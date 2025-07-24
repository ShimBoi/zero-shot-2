import improve.custom.models.rt1 as rt1
from improve.custom.models.rt1_actor_critic import RT1ActorCritic


def createActorCriticModel(checkpoint_path=None):
    model = rt1.RT1(
        num_image_tokens=81,
        num_action_tokens=11,
        layer_size=256,
        vocab_size=512,
        use_token_learner=True,
        world_vector_range=(-2.0, 2.0)
    )

    model_rl = RT1ActorCritic(
        checkpoint_path=checkpoint_path,
        model=model,
    )

    return model_rl


if __name__ == "__main__":
    model_rl = createActorCriticModel("improve/custom/models/rt_1_x_jax/")
    print(model_rl)
