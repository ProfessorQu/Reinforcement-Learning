import os
import sys
import tensorflow as tf
# cannot import name 'ResponseError' from 'minio.error'
from rl_coach.coach import CoachInterface

coach = CoachInterface(preset='CartPole_ClippedPPO',
                       custom_parameter='heatup_steps=EnvironmentSteps(5);'
                       'improve_steps=TrainingSteps(3)',
                       num_workers=1, checkpoint_save_secs=10)
coach.run()
