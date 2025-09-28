"""
Interactive Threshold Regression Model Creator for Confocal Images

This tool provides an interactive interface for:
1. Loading confocal images
2. Setting thresholds manually with a slider
3. Creating a regression model from the collected thresholds
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random
from aicsimageio import AICSImage

class InteractiveRegressionTrainer:
    def __init__(self, input_dir, num_files, min_threshold=0, max_threshold=500, threshold_step=1.0, metric_percentile=15, use_replicates=True):
        self.input_dir = input_dir
        self.num_files = num_files
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_step = threshold_step
        self.metric_percentile = metric_percentile
        self.use_replicates = use_replicates
        
        # State variables
        self.images = []
        self.image_paths = []
        self.current_image_idx = 0
        self.l1cam_thresholds = {}
        self.image_metrics = {}
        self.skipped_images = set()
        
        # Load images
        self._load_images()
        
    def _load_images(self):
        """Load images from directory structure"""
        print(f"Loading {self.num_files} images...")
        
        # Find all .nd2 files recursively
        all_images = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.nd2'):
                    all_images.append(os.path.join(root, file))
        
        if not all_images:
            raise ValueError(f"No .nd2 files found in {self.input_dir}")
        
        # Group files by replicate if using replicates
        if self.use_replicates:
            replicate_files = {'B114': [], 'B115': [], 'B116': [], 'B117': []}
            for file_path in all_images:
                for rep in replicate_files:
                    if rep in file_path:
                        replicate_files[rep].append(file_path)
                        break
            
            # Sample evenly from each replicate
            selected_files = []
            files_per_replicate = max(1, self.num_files // len(replicate_files))
            for rep, files in replicate_files.items():
                if files:  # Only sample if replicate has files
                    sample_size = min(files_per_replicate, len(files))
                    selected_files.extend(random.sample(files, sample_size))
            
            # If we need more files to reach num_files, sample randomly from all remaining
            remaining_needed = self.num_files - len(selected_files)
            if remaining_needed > 0:
                remaining_files = [f for f in all_images if f not in selected_files]
                if remaining_files:
                    selected_files.extend(random.sample(remaining_files, min(remaining_needed, len(remaining_files))))
        else:
            # Random sampling without considering replicates
            selected_files = random.sample(all_images, min(self.num_files, len(all_images)))
        
        print(f"Loading {len(selected_files)} images...")
        
        # Load each image with timeout and retry
        from functools import partial
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Image loading timed out")
        
        # Set timeout for image loading (10 seconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        
        for file_path in selected_files:
            retries = 3
            while retries > 0:
                try:
                    signal.alarm(10)  # Set timeout
                    
                    # Load image data
                    img = AICSImage(file_path)
                    
                    # Get L1CAM (FITC, C=0) channel
                    l1cam_data = img.get_image_data("ZYX", C=0)
                    
                    signal.alarm(0)  # Disable timeout
                    
                    # Convert to max intensity projection if 3D
                    if len(l1cam_data.shape) == 3:
                        l1cam_data = np.max(l1cam_data, axis=0)
                    
                    # Calculate image metrics
                    vals = l1cam_data.ravel()
                    thr = np.percentile(vals, self.metric_percentile)
                    above_thr = vals[vals > thr]
                    mean_above = above_thr.mean() if above_thr.size > 0 else 0.0
                    
                    # Store image data
                    self.images.append(l1cam_data)
                    self.image_paths.append(file_path)
                    self.image_metrics[file_path] = mean_above
                    self.l1cam_thresholds[file_path] = None
                    break  # Success, exit retry loop
                    
                except TimeoutError:
                    print(f"Timeout loading {os.path.basename(file_path)}, {retries-1} retries left")
                    retries -= 1
                except Exception as e:
                    print(f"Error loading {os.path.basename(file_path)}: {str(e)}, {retries-1} retries left")
                    retries -= 1
                finally:
                    signal.alarm(0)  # Ensure timeout is disabled
            
            if retries == 0:
                print(f"Failed to load {os.path.basename(file_path)} after 3 attempts")
        
        if not self.images:
            raise RuntimeError("Failed to load any images successfully")
        
        print(f"Successfully loaded {len(self.images)} images")
        
        # Print replicate distribution if using replicates
        if self.use_replicates:
            rep_counts = {}
            for path in self.image_paths:
                for rep in ['B114', 'B115', 'B116', 'B117']:
                    if rep in path:
                        rep_counts[rep] = rep_counts.get(rep, 0) + 1
                        break
            print("\nReplicate distribution:")
            for rep, count in sorted(rep_counts.items()):
                print(f"{rep}: {count} images")
    
    def _update_display(self):
        """Update the display with current image and controls"""
        if not self.images:
            print("No images loaded!")
            return
        
        # Create the display elements
        current_path = self.image_paths[self.current_image_idx]
        current_image = self.images[self.current_image_idx]
        current_metric = self.image_metrics[current_path]
        current_threshold = self.l1cam_thresholds.get(current_path, self.min_threshold)
        
        # Create binary mask
        binary_mask = current_image > (current_threshold if current_threshold is not None else self.min_threshold)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original image
        ax1.imshow(current_image, cmap='gray')
        ax1.set_title('L1CAM Image')
        ax1.axis('off')
        
        # Plot binary mask
        ax2.imshow(binary_mask, cmap='gray')
        ax2.set_title(f'Binary Mask (Threshold: {current_threshold if current_threshold is not None else "Not Set"})')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Create widgets
        slider = widgets.FloatSlider(
            value=current_threshold if current_threshold is not None else self.min_threshold,
            min=self.min_threshold,
            max=self.max_threshold,
            step=self.threshold_step,
            description='Threshold:',
            continuous_update=True
        )
        
        prev_button = widgets.Button(description='Previous')
        next_button = widgets.Button(description='Next')
        finish_button = widgets.Button(description='Finish Now')
        skip_button = widgets.Button(description='Skip')
        
        # Navigation info
        progress_text = f"Image {self.current_image_idx + 1}/{len(self.images)}"
        if current_path in self.skipped_images:
            progress_text += " (Skipped)"
        progress = widgets.HTML(value=f"<b>{progress_text}</b>")
        
        # Image info
        image_name = os.path.basename(current_path)
        replicate = "Unknown"
        if self.use_replicates:
            for rep in ['B114', 'B115', 'B116', 'B117']:
                if rep in current_path:
                    replicate = rep
                    break
        info_text = f"""
        <b>Image info:</b><br>
        File: {image_name}<br>
        Replicate: {replicate}<br>
        Mean above {self.metric_percentile}th percentile: {current_metric:.2f}
        """
        info = widgets.HTML(value=info_text)
        
        # Layout
        button_box = widgets.HBox([prev_button, next_button, skip_button, finish_button])
        display(widgets.VBox([progress, info, slider, button_box]))
        
        def on_threshold_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # Update threshold
                new_threshold = change['new']
                self.l1cam_thresholds[current_path] = new_threshold
                
                # Update binary mask
                binary_mask = current_image > new_threshold
                ax2.clear()
                ax2.imshow(binary_mask, cmap='gray')
                ax2.set_title(f'Binary Mask (Threshold: {new_threshold:.1f})')
                ax2.axis('off')
                fig.canvas.draw_idle()
        
        def on_prev_click(b):
            if self.current_image_idx > 0:
                self.current_image_idx -= 1
                plt.close(fig)
                clear_output(wait=True)
                self._update_display()
        
        def on_next_click(b):
            if self.current_image_idx < len(self.images) - 1:
                self.current_image_idx += 1
                plt.close(fig)
                clear_output(wait=True)
                self._update_display()
        
        def on_skip_click(b):
            self.skipped_images.add(current_path)
            if self.current_image_idx < len(self.images) - 1:
                self.current_image_idx += 1
                plt.close(fig)
                clear_output(wait=True)
                self._update_display()
            else:
                self._create_and_save_model()
        
        def on_finish_click(b):
            plt.close(fig)
            clear_output(wait=True)
            self._create_and_save_model()
        
        slider.observe(on_threshold_change, names='value')
        prev_button.on_click(on_prev_click)
        next_button.on_click(on_next_click)
        skip_button.on_click(on_skip_click)
        finish_button.on_click(on_finish_click)
    
    def _create_and_save_model(self):
        """Create regression model from collected thresholds"""
        # Filter out skipped images
        X = []  # metrics
        y = []  # thresholds
        replicates = []  # replicate identifiers
        
        for path, threshold in self.l1cam_thresholds.items():
            if path not in self.skipped_images and threshold is not None:
                X.append([self.image_metrics[path]])
                y.append(threshold)
                if self.use_replicates:
                    rep = "Unknown"
                    for r in ['B114', 'B115', 'B116', 'B117']:
                        if r in path:
                            rep = r
                            break
                    replicates.append(rep)
        
        if not X:
            print("No valid thresholds collected!")
            return
        
        # Create and fit model
        X = np.array(X)
        y = np.array(y)
        
        # Create model data
        model_data = {
            'metric_percentile': self.metric_percentile,
            'created_at': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'num_training_samples': len(X),
            'use_replicates': self.use_replicates
        }
        
        if self.use_replicates:
            # Calculate replicate offsets
            unique_reps = sorted(set(replicates))
            rep_offsets = {}
            base_predictions = []
            
            # First, fit model without replicate offsets
            base_model = LinearRegression()
            base_model.fit(X, y)
            model_data['intercept'] = float(base_model.intercept_)
            model_data['metric_coefficient'] = float(base_model.coef_[0])
            
            # Calculate base predictions
            base_predictions = base_model.predict(X)
            
            # Calculate offsets for each replicate
            for rep in unique_reps:
                rep_mask = np.array(replicates) == rep
                if np.any(rep_mask):
                    offset = np.mean(y[rep_mask] - base_predictions[rep_mask])
                    rep_offsets[rep] = float(offset)
            
            model_data['replicate_offsets'] = rep_offsets
            
            # Calculate R² score with replicate offsets
            y_pred = base_predictions.copy()
            for i, rep in enumerate(replicates):
                y_pred[i] += rep_offsets.get(rep, 0.0)
            r2 = r2_score(y, y_pred)
            model_data['r2_score'] = float(r2)
            
        else:
            # Simple linear regression without replicate offsets
            model = LinearRegression()
            model.fit(X, y)
            model_data.update({
                'intercept': float(model.intercept_),
                'metric_coefficient': float(model.coef_[0]),
                'replicate_offsets': {},
                'r2_score': float(r2_score(y, model.predict(X)))
            })
        
        # Save model
        models_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Save model as JSON
        json_path = os.path.join(models_dir, f'confocal_regression_model_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        # Create documentation
        doc = {
            'Model Type': 'Confocal Threshold Regression',
            'Created At': model_data['created_at'],
            'Training Samples': model_data['num_training_samples'],
            'R² Score': f"{model_data['r2_score']:.4f}",
            'Parameters': {
                'Intercept': f"{model_data['intercept']:.4f}",
                'Metric Coefficient': f"{model_data['metric_coefficient']:.4f}",
                'Metric Percentile': model_data['metric_percentile'],
                'Use Replicates': str(model_data['use_replicates'])
            }
        }
        
        if self.use_replicates:
            doc['Replicate Offsets'] = {rep: f"{offset:.4f}" 
                                      for rep, offset in model_data['replicate_offsets'].items()}
        
        # Save documentation
        txt_path = os.path.join(models_dir, f'confocal_regression_model_{timestamp}.txt')
        with open(txt_path, 'w') as f:
            for key, value in doc.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        if self.use_replicates:
            # Plot points with different colors for each replicate
            for rep in unique_reps:
                mask = np.array(replicates) == rep
                plt.scatter(X[mask], y[mask], alpha=0.5, label=f'Replicate {rep}')
            
            # Plot regression line with offsets
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            base_pred = model_data['intercept'] + model_data['metric_coefficient'] * x_range.flatten()
            plt.plot(x_range, base_pred, 'k--', label='Base regression', alpha=0.5)
            
            for rep in unique_reps:
                offset = model_data['replicate_offsets'].get(rep, 0.0)
                plt.plot(x_range, base_pred + offset, '-', label=f'{rep} adjusted', alpha=0.7)
        else:
            plt.scatter(X, y, alpha=0.5, label='Training Data')
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = model_data['intercept'] + model_data['metric_coefficient'] * x_range.flatten()
            plt.plot(x_range, y_pred, 'r-', label='Regression Line')
        
        plt.xlabel(f'Mean Above {self.metric_percentile}th Percentile')
        plt.ylabel('Threshold')
        plt.title('Threshold Regression Model')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(models_dir, f'confocal_regression_plot_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print success message
        print("✨ Regression model created successfully!")
        print("\nFiles created:")
        print(f"• Model: {os.path.basename(json_path)}")
        print(f"• Documentation: {os.path.basename(txt_path)}")
        print(f"• Plot: {os.path.basename(plot_path)}")
        print(f"\nLocation: {models_dir}")
        
        print("\nTo use this model in the analysis:")
        print("1. Copy the .json file to the Active_Model_and_Configuration_Directory")
        print("2. Update config.yaml with:")
        print("   use_regression_model: true")
        print("   use_raw_threshold: false")
    
    def run(self):
        """Run the interactive training interface"""
        if not self.images:
            print("No images loaded!")
            return
        
        print("\n===============================================================================")
        print("MANUAL THRESHOLD SETTING")
        print("===============================================================================")
        print("You will see each image and manually set the threshold value.")
        print("Look at the L1CAM image and decide what threshold best captures the signal.")
        print("===============================================================================\n")
        
        self._update_display()

def create_interactive_model(input_dir, num_files=10, min_threshold=0, max_threshold=500, threshold_step=1.0, metric_percentile=15, use_replicates=True):
    """
    Create an interactive regression model training interface.
    
    Args:
        input_dir (str): Path to input directory containing images
        num_files (int): Number of random files to use for training
        min_threshold (float): Minimum threshold value
        max_threshold (float): Maximum threshold value
        threshold_step (float): Step size for threshold slider
        metric_percentile (float): Percentile for calculating mean above value
        use_replicates (bool): Whether to use replicate-specific offsets
    
    Returns:
        None (saves model interactively)
    """
    print("Starting interactive regression model creation...")
    print(f"Will sample {num_files} random files from: {input_dir}")
    print(f"Threshold range: {min_threshold} to {max_threshold} (step: {threshold_step})")
    print(f"Using replicate offsets: {use_replicates}")
    print("Use the slider to adjust threshold and click 'Finish Now' when satisfied.")
    
    trainer = InteractiveRegressionTrainer(
        input_dir,
        num_files,
        min_threshold,
        max_threshold,
        threshold_step,
        metric_percentile,
        use_replicates
    )
    trainer.run()