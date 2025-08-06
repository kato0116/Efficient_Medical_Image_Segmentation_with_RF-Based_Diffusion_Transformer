import guided_diffusion.gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from rectified_flow.rectified_flow import RectifiedFlow


def select_type(type, args):
    if   type == "diffusion":
        print("Diffusion Model")
        return create_gaussian_diffusion(
            steps=1000,
            learn_sigma=args.learn_sigma,
            sigma_small=False,
            noise_schedule=args.noise_schedule,
            use_kl=False, # 損失でKL使うか
            predict_xstart=args.predict_xstart,
            rescale_timesteps=False,
            rescale_learned_sigmas=False, # 
            timestep_respacing=args.ddim,
        )
    elif type == "rectified_flow":
        print("Rectified Flow")
        return create_rectified_flow(
            init_type='gaussian', 
            time_sampler=args.time_sampler,
            noise_scale=1.0, 
            use_ode_sampler='euler', 
            sigma_var=0.0, 
            ode_tol=1e-5, 
            sample_N=args.euler_step,
            euler_ensemble=args.euler_ensemble,
            sampler_type=args.sampler_type,
        )
    elif type == "flow_matching":
        print("Flow Matching")
        return create_flow_matching()
    
    else:
        print(f"Unknown type: {type}")
        return None
    
def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False, # 損失でKL使うか
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False, # 
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type=gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type=gd.LossType.RESCALED_MSE
    else:
        loss_type=gd.LossType.MSE
        
    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas = betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def create_rectified_flow(
    init_type='gaussian',
    time_sampler="uniform",
    noise_scale=1.0, 
    use_ode_sampler='euler', 
    sigma_var=0.0, 
    ode_tol=1e-5, 
    sample_N=None,
    euler_ensemble=0,
    sampler_type='euler',
    ):
    return RectifiedFlow(
        init_type=init_type,
        time_sampler=time_sampler,
        noise_scale=noise_scale,
        use_ode_sampler=use_ode_sampler,
        sigma_var=sigma_var,
        ode_tol=ode_tol,
        sample_N=sample_N,
        euler_ensemble=euler_ensemble,
        sampler_type=sampler_type,
    )

# 未実装
def create_flow_matching():
    return None
