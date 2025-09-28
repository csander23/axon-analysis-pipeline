"""
Interactive Threshold Regression Model Creator for Slide Scanning Images

This tool provides an interactive interface for:
1. Loading slide scanning images
2. Setting thresholds manually with a slider
3. Creating a regression model from the collected thresholds
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Image as IPImage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random
import cv2
import base64
from io import BytesIO

class InteractiveRegressionTrainer:
    def __init__(self, input_dir, num_files, min_threshold=0.010, max_threshold=0.050, threshold_step=0.001, metric_percentile=None, use_replicates=True):
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
        
        all_images = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.tif', '.tiff')):
                    all_images.append(os.path.join(root, file))
        
        # Sample images across all bioreplicates if using replicates
        if self.use_replicates:
            biorep_images = {}
            for img_path in all_images:
                biorep = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                if biorep not in biorep_images:
                    biorep_images[biorep] = []
                biorep_images[biorep].append(img_path)
            
            # Select images evenly from each bioreplicate
            selected_images = []
            while len(selected_images) < self.num_files and biorep_images:
                for biorep in list(biorep_images.keys()):
                    if biorep_images[biorep]:
                        img = random.choice(biorep_images[biorep])
                        selected_images.append(img)
                        biorep_images[biorep].remove(img)
                        if not biorep_images[biorep]:
                            del biorep_images[biorep]
                    if len(selected_images) >= self.num_files:
                        break
        else:
            # Random sampling without considering replicates
            selected_images = random.sample(all_images, min(self.num_files, len(all_images)))
        
        # Load selected images
        for img_path in selected_images:
            try:
                # Load image and convert to grayscale
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert to grayscale and normalize to 0-1 range (legacy method)
                if len(img.shape) == 3:
                    gray = np.mean(img, axis=2)  # Mean of all channels
                else:
                    gray = img
                gray = gray.astype(np.float32) / 255.0  # Normalize to 0-1
                
                # Calculate whole image mean (legacy method)
                mean_above = float(np.mean(gray))
                
                self.images.append(gray)  # Store normalized grayscale image
                self.image_paths.append(img_path)
                self.image_metrics[img_path] = mean_above
                self.l1cam_thresholds[img_path] = self.min_threshold  # Initialize with minimum threshold
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"Successfully loaded {len(self.images)} images")
        
        # Print replicate distribution if using replicates
        if self.use_replicates:
            biorep_counts = {}
            for path in self.image_paths:
                biorep = os.path.basename(os.path.dirname(os.path.dirname(path)))
                biorep_counts[biorep] = biorep_counts.get(biorep, 0) + 1
            print("\nReplicate distribution:")
            for biorep, count in sorted(biorep_counts.items()):
                print(f"{biorep}: {count} images")
    
    def _update_display(self):
        """Update the display with current image and controls"""
        if not self.images:
            print("No images loaded!")
            return
        
        # Create the display elements
        current_path = self.image_paths[self.current_image_idx]
        current_image = self.images[self.current_image_idx]
        current_metric = self.image_metrics[current_path]
        current_threshold = self.l1cam_thresholds.get(current_path)
        if current_threshold is None:
            current_threshold = self.min_threshold
            self.l1cam_thresholds[current_path] = current_threshold
        
        # Current image is already normalized grayscale
        gray = current_image
        
        # Create binary mask
        binary_mask = gray > current_threshold
        
        # Clear previous output
        clear_output(wait=True)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original image
        ax1.imshow(gray, cmap='gray')
        ax1.set_title('L1CAM Image')
        ax1.axis('off')
        
        # Plot binary mask
        ax2.imshow(binary_mask, cmap='gray')
        ax2.set_title(f'Binary Mask (Threshold: {current_threshold:.3f})')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Create widgets
        slider = widgets.FloatSlider(
            value=current_threshold,
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
        replicate = os.path.basename(os.path.dirname(os.path.dirname(current_path))) if self.use_replicates else "N/A"
        info_text = f"""
        <b>Image info:</b><br>
        File: {image_name}<br>
        Replicate: {replicate}<br>
        Mean: {current_metric:.3f}
        """
        info = widgets.HTML(value=info_text)
        
        # Convert figure to base64 for display
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        buf.close()
        plt.close(fig)
        
        # Create HTML image display
        img_html = f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'
        image_display = widgets.HTML(value=img_html)
        
        # Layout
        button_box = widgets.HBox([prev_button, next_button, skip_button, finish_button])
        
        # Display everything
        display(widgets.VBox([progress, info, slider, button_box, image_display]))
        
        def on_threshold_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                new_threshold = change['new']
                self.l1cam_thresholds[current_path] = new_threshold
                
                # Create new figure with updated threshold
                new_fig, (new_ax1, new_ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot original image
                new_ax1.imshow(gray, cmap='gray')
                new_ax1.set_title('L1CAM Image')
                new_ax1.axis('off')
                
                # Plot updated binary mask
                binary_mask = gray > new_threshold
                new_ax2.imshow(binary_mask, cmap='gray')
                new_ax2.set_title(f'Binary Mask (Threshold: {new_threshold:.3f})')
                new_ax2.axis('off')
                
                plt.tight_layout()
                
                # Convert to base64 and update display
                buf = BytesIO()
                new_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode()
                buf.close()
                plt.close(new_fig)
                
                # Update image display
                img_html = f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'
                image_display.value = img_html
        
        def on_prev_click(b):
            if self.current_image_idx > 0:
                self.current_image_idx -= 1
                self._update_display()
        
        def on_next_click(b):
            if self.current_image_idx < len(self.images) - 1:
                self.current_image_idx += 1
                self._update_display()
        
        def on_skip_click(b):
            self.skipped_images.add(current_path)
            if self.current_image_idx < len(self.images) - 1:
                self.current_image_idx += 1
                self._update_display()
            else:
                self._create_and_save_model()
        
        def on_finish_click(b):
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
                    replicate = os.path.basename(os.path.dirname(os.path.dirname(path)))
                    replicates.append(replicate)
        
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
        json_path = os.path.join(models_dir, f'slide_scanning_regression_model_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        # Create documentation
        doc = {
            'Model Type': 'Slide Scanning Threshold Regression',
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
        txt_path = os.path.join(models_dir, f'slide_scanning_regression_model_{timestamp}.txt')
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
        
        plt.xlabel('Mean Intensity')
        plt.ylabel('Threshold')
        plt.title('Threshold Regression Model')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(models_dir, f'slide_scanning_regression_plot_{timestamp}.png')
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

def create_interactive_model(input_dir, num_files=10, min_threshold=0.010, max_threshold=0.050, threshold_step=0.001, metric_percentile=None, use_replicates=True):
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