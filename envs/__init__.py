from gym.envs.registration import register

from envs.curriculum_env import CurriculumFishingEnv

register(
    id="curriculum_fishing-v0",
    entry_point="envs:CurriculumFishingEnv",
)
