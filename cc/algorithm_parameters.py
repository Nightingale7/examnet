ogan_parameters = {"fitness_coef": 0.95,
                   "train_delay": 1,
                   "N_candidate_tests": 1,
                   "reset_each_training": True
                   }

ogan_model_parameters = {
    "convolution": {
        "optimizer": "Adam",
        "discriminator_lr": 0.001,
        "discriminator_betas": [0.9, 0.999],
        "generator_lr": 0.0001,
        "generator_betas": [0.9, 0.999],
        "noise_batch_size": 8192,
        "generator_loss": "MSE,Logit",
        "discriminator_loss": "MSE,Logit",
        "generator_mlm": "GeneratorNetwork",
        "generator_mlm_parameters": {
            "noise_dim": 20,
            "hidden_neurons": [128,128,128],
            "hidden_activation": "leaky_relu"
        },
        "discriminator_mlm": "DiscriminatorNetwork1dConv",
        "discriminator_mlm_parameters": {
            "feature_maps": [16, 16],
            "kernel_sizes": [[2,2], [2,2]],
            "convolution_activation": "relu",
            "convolution_activation": "leaky_relu",
            "dense_neurons": 128
        },
        "train_settings_init": {"epochs": 1, "discriminator_epochs": 15, "generator_batch_size": 32},
        "train_settings": {"epochs": 1, "discriminator_epochs": 15, "generator_batch_size": 32}
    }
}

wogan_parameters = {
    "wgan_batch_size": 32,
    "fitness_coef": 0.95,
    "train_delay": 3,
    "N_candidate_tests": 1,
    "sampler_id": "SBST_Sampler",
    "sampler_parameters": {
        "shift_function": "linear",
        "bins": 10,
        "sampling_bins": 10,
        "sample_with_replacement": False,
        "omit_initial_empty": False,
        "quantile_start": 1/3,
        "quantile_end": 1/3,
        "zero_minimum": True,
        "shift_function_parameters": {"initial": 0, "final": 3}
    }
}

analyzer_mlm_parameters = {
    "convolution": {
        "feature_maps": [16,16],
        "kernel_sizes": [[2,2], [2,2]],
        "convolution_activation": "leaky_relu",
        "dense_neurons": 128
    }
}

wogan_model_parameters = {
    "critic_optimizer": "Adam",
    "critic_lr": 0.0001,
    "critic_betas": [0, 0.9],
    "generator_optimizer": "Adam",
    "generator_lr": 0.0001,
    "generator_betas": [0, 0.9],
    "noise_batch_size": 32,
    "gp_coefficient": 10,
    "eps": 1e-6,
    "report_wd": True,
    "analyzer": "Analyzer_NN",
    "analyzer_parameters": {
        "optimizer": "Adam",
        "lr": 0.001,
        "betas": [0, 0.9],
        "loss": "MSE,logit",
        "l2_regularization_coef": 0.01,
        "analyzer_mlm": "AnalyzerNetwork_conv",
        "analyzer_mlm_parameters": analyzer_mlm_parameters["convolution"]
    },
    "generator_mlm": "GeneratorNetwork",
    "generator_mlm_parameters": {
        "noise_dim": 10,
        "hidden_neurons": [128,128],
        "hidden_activation": "relu",
        "batch_normalization": True,
        "layer_normalization": False
    },
    "critic_mlm": "CriticNetwork",
    "critic_mlm_parameters": {
        "hidden_neurons": [128,128],
        "hidden_activation": "leaky_relu",
    },
    "train_settings_init": {
        "epochs": 3,
        "analyzer_epochs": 20,
        "critic_steps": 5,
        "generator_steps": 1
    },
    "train_settings": {
        "epochs": 2,
        "analyzer_epochs": 10,
        "critic_steps": 5,
        "generator_steps": 1
    },
}

diffusion_parameters = {
    "fitness_coef":               0.95,
    "train_delay":                1,
    "invalid_threshold":          100,
    "N_candidate_tests":          5,
    "tmp":                        0.0,
    "exploration_threshold":      0.1,
    "exploration_duration":       0.05,
    "exploration_probability":    0.5,
    "exploration_selection":      "random", # best, random
    "exploration_var_multiplier": 1,
    "sampler_id":                 "Quantile_Sampler",
    "sampler_parameters": {
        "bins":                    10,
        "sample_with_replacement": False,
        "quantile_start":          0.4,
        "quantile_end":            0.03,
        "zero_minimum":            True
    },
    "train_settings": {
        "diffusion": {
            "epochs":     20,
            "batch_size": 32
        }
    }
}

diffusion_model_parameters = {
    "analyzer": "RandomForest",
    "analyzer_parameters": {
    },
    "backward_model": "UNet",
    "backward_model_parameters": {
        "time_embedding_dim":  20,
        "residual_connection": True,
        "max_depth":           5
    },
    "ddpm": "DDPM",
    "ddpm_parameters": {
        "N_steps":  75,
        "min_beta": 10**(-4),
        "max_beta": 0.02
    },
    "ddpm_optimizer_parameters": {
        "optimizer": "Adam",
        "lr":        0.001
    }
}

