datasets=("FMNIST" "MNIST" "KMNIST")
models=("DDIM" "PF")
methods=("F-PNDM" "PF" "PR-CR" "DDIM" "S-PNDM" "SP-PNDM")
diffusion_steps=("600" "800" "1000")

# Iterate through all combinations
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            for diffusion_step in "${diffusion_steps[@]}"; do
                ckpt="/home/ubuntu/derek-240306/jxu/PNDM/DDIM_and_PF_on_MNIST_FMNIST_KMNIST/${model}-${dataset}/save_ckpt/train.ckpt";
                output_path="/home/ubuntu/derek-240306/jxu/PNDM/inference_results/${dataset}/${model}-${method}/${diffusion_step}";
                python run.py sample.model_path=$ckpt model=$model \
                    sample_speed=100 dataset.dataset=$dataset method=$method scheduler.diffusion_step=$diffusion_step \
                    sample.image_output_path=$output_path;
            done
        done
    done
done