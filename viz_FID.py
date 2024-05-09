import os, json
import matplotlib.pyplot as plt

base_dir = 'inference_results'

diffusion_steps = [600, 800, 1000]

def plot_fid_methods(model_data, model_name, dataset_name):
    plt.figure(figsize=(10, 5))
    for method, values in model_data.items():
        fid_values = [values[str(step)] for step in diffusion_steps]
        plt.plot(diffusion_steps, fid_values, marker='o', linestyle='-', label=method)
    plt.title(f'FID for {model_name} Model Across Different Diffusion Steps on {dataset_name}')
    plt.xlabel('Diffusion Steps')
    plt.ylabel('FID')
    plt.legend()
    plt.grid(True)
    plt.show()

for dataset in ['MNIST', 'KMNIST', 'FMNIST']:
    fid_result_path = os.path.join(
        base_dir, dataset, 'fid_results.json'
    )
    with open(fid_result_path, 'r') as f:
        data: dict = json.load(f)
    
    plot_fid_methods(data['DDIM'], "DDIM", dataset)
    plt.savefig(os.path.join(base_dir, dataset, f'fid_{dataset}_DDIM.png'), bbox_inches='tight', dpi=400, pad_inches=0.1)

    plot_fid_methods(data['PF'], "PF", dataset)
    plt.savefig(os.path.join(base_dir, dataset, f'fid_{dataset}_PF.png'), bbox_inches='tight', dpi=400, pad_inches=0.1)