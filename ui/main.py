#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path

# Add script directory to Python path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from ui.memory_sam_ui import MemorySAMUI

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Enhanced Memory SAM Interface")
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="hiera_l", 
        choices=["hiera_b+", "hiera_l", "hiera_s", "hiera_t"],
        help="SAM2 model type to use"
    )
    
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=None,
        help="SAM2 checkpoint path (default: auto-detect)"
    )
    
    parser.add_argument(
        "--dinov2_model", 
        type=str, 
        default="facebook/dinov2-base",
        help="DINOv2 model to use"
    )
    
    parser.add_argument(
        "--memory_dir", 
        type=str, 
        default="memory",
        help="Memory system directory"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results",
        help="Results directory"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda, cpu, or mps)"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Share Gradio interface with public URL"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.memory_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize Memory SAM UI
    memory_ui = MemorySAMUI(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        dinov2_model=args.dinov2_model,
        memory_dir=args.memory_dir,
        results_dir=args.results_dir,
        device=args.device,
        use_sparse_matching=True
    )
    
    # Set up and run Gradio interface
    app = memory_ui.setup_interface()
    
    # Add memory and results directory paths to allowed_paths
    memory_path = Path(args.memory_dir).resolve()
    results_path = Path(args.results_dir).resolve()
    
    # Add all item directory paths in memory directory
    allowed_paths = [str(memory_path), str(results_path)]
    
    # Add all subdirectories in memory directory
    for item_dir in memory_path.glob("item_*"):
        if item_dir.is_dir():
            allowed_paths.append(str(item_dir))
    
    # Add all subdirectories in results directory
    for result_dir in results_path.glob("*"):
        if result_dir.is_dir():
            allowed_paths.append(str(result_dir))
    
    print(f"Allowed paths: {allowed_paths}")
    app.launch(share=args.share, allowed_paths=allowed_paths)
    
    print("Interface closed.")

if __name__ == "__main__":
    main()