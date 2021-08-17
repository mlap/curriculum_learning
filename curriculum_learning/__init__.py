from gym.envs.registration import register
__version__ = "0.0.0"

entry_point = "curriculum_learning.envs:CurriculumFishingEnv"
register(
    id="curriculum_fishing-v0",
    entry_point=entry_point,
)
