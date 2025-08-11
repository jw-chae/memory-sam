import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import MemorySAMPredictor
from scripts.memory_sam_predictor import MemorySAMPredictor
from scripts.memory_sam_predictor_backup import MemorySAMPredictor as MemorySAMPredictorOriginal

def parse_args():
    parser = argparse.ArgumentParser(description='Memory SAM with Sparse Matching Example')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--reference', type=str, default=None, help='Optional reference image path')
    parser.add_argument('--method', type=str, default='sparse', choices=['sparse', 'global', 'both'], 
                        help='Matching method to use')
    parser.add_argument('--model_type', type=str, default='hiera_l', 
                        help='SAM2 model type (hiera_b+, hiera_l, hiera_s, hiera_t)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--memory_dir', type=str, default='memory', help='Memory directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    results = {}
    
    # Create output directory
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process with the selected method(s)
    if args.method in ['sparse', 'both']:
        print("\n--- Processing with Sparse Matching ---")
        predictor_sparse = MemorySAMPredictor(
            model_type=args.model_type,
            device=args.device,
            memory_dir=args.memory_dir,
            results_dir=args.results_dir,
            use_sparse_matching=True
        )
        
        result_sparse = predictor_sparse.process_image(
            image_path=args.image,
            reference_path=args.reference,
            use_sparse_matching=True
        )
        
        results['sparse'] = result_sparse
        print(f"Result path: {result_sparse['result_path']}")
        
        # Save for comparison
        Image.fromarray(result_sparse['image']).save(output_dir / 'input.png')
        mask_img = (result_sparse['mask'] * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(output_dir / 'sparse_mask.png')
        Image.fromarray(predictor_sparse.visualize_mask(result_sparse['image'], result_sparse['mask'])).save(output_dir / 'sparse_visualization.png')
    
    if args.method in ['global', 'both']:
        print("\n--- Processing with Global Matching ---")
        # Load the original (backup) predictor without sparse matching
        try:
            predictor_global = MemorySAMPredictorOriginal(
                model_type=args.model_type,
                device=args.device,
                memory_dir=args.memory_dir,
                results_dir=args.results_dir
            )
            
            result_global = predictor_global.process_image(
                image_path=args.image,
                reference_path=args.reference
            )
            
            results['global'] = result_global
            print(f"Result path: {result_global['result_path']}")
            
            # Save for comparison
            mask_img = (result_global['mask'] * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(output_dir / 'global_mask.png')
            Image.fromarray(predictor_global.visualize_mask(result_global['image'], result_global['mask'])).save(output_dir / 'global_visualization.png')
        except Exception as e:
            print(f"Error running global matching: {e}")
            # Fallback to the new predictor with sparse matching disabled
            predictor_global = MemorySAMPredictor(
                model_type=args.model_type,
                device=args.device,
                memory_dir=args.memory_dir,
                results_dir=args.results_dir,
                use_sparse_matching=False
            )
            
            result_global = predictor_global.process_image(
                image_path=args.image,
                reference_path=args.reference,
                use_sparse_matching=False
            )
            
            results['global'] = result_global
            print(f"Result path: {result_global['result_path']}")
            
            # Save for comparison
            mask_img = (result_global['mask'] * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(output_dir / 'global_mask.png')
            Image.fromarray(predictor_global.visualize_mask(result_global['image'], result_global['mask'])).save(output_dir / 'global_visualization.png')
    
    # Create comparison visualization if both methods were used
    if args.method == 'both' and 'sparse' in results and 'global' in results:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axs[0].imshow(results['sparse']['image'])
        axs[0].set_title('Input Image')
        axs[0].axis('off')
        
        # Global matching result
        axs[1].imshow(predictor_global.visualize_mask(results['global']['image'], results['global']['mask']))
        axs[1].set_title(f'Global Matching (score: {results["global"]["score"]:.3f})')
        axs[1].axis('off')
        
        # Sparse matching result
        axs[2].imshow(predictor_sparse.visualize_mask(results['sparse']['image'], results['sparse']['mask']))
        axs[2].set_title(f'Sparse Matching (score: {results["sparse"]["score"]:.3f})')
        axs[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison.png', dpi=300)
        print(f"Comparison saved to {output_dir / 'comparison.png'}")
        
        # If sparse matching is available, create visualization of patch matches
        if hasattr(predictor_sparse, 'visualize_sparse_matches') and len(results['sparse']['similar_items']) > 0:
            try:
                best_match_id = results['sparse']['similar_items'][0]['item']['id']
                best_match_data = predictor_sparse.memory.load_item_data(best_match_id)
                
                match_vis = predictor_sparse.visualize_sparse_matches(
                    best_match_data['image'], 
                    results['sparse']['image'],
                    best_match_data['mask'],
                    results['sparse']['mask']
                )
                
                plt.figure(figsize=(15, 10))
                plt.imshow(match_vis)
                plt.title('Sparse Feature Matches')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_dir / 'feature_matches.png', dpi=300)
                print(f"Feature matches visualization saved to {output_dir / 'feature_matches.png'}")
            except Exception as e:
                print(f"Could not create feature matches visualization: {e}")
    
    print(f"\nProcessing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()