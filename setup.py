from setuptools import setup

setup(
    name='rl4mm',
    version='0.1.0',
    packages=['rl4mm', 'rl4mm.gym', 'rl4mm.gym.tests', 'rl4mm.gym.order_tracking', 'rl4mm.gym.action_interpretation',
              'rl4mm.utils', 'rl4mm.agents', 'rl4mm.extras', 'rl4mm.helpers', 'rl4mm.rewards', 'rl4mm.database',
              'rl4mm.database.tests', 'rl4mm.features', 'rl4mm.features.tests', 'rl4mm.orderbook',
              'rl4mm.orderbook.tests', 'rl4mm.simulation', 'rl4mm.simulation.tests'],
    url='https://github.com/jjjerome/RL4MM',
    license='3-Clause BSD License',
    author='jjjerome',
    author_email='josephjerome94@gmail.com',
    description='reinforcement learning for market making'
)
