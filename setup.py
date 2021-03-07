from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='phd-satellite-trajectory',
      version='0.1+snapshot',
      python_requires='>=3.6',
      packages=find_packages(where='.', exclude=('test',)),
      scripts=['scripts/launch_tensorboard.sh'],
      entry_points={"console_scripts": ["perigeeraising_continuous1d_a2c = phd_satellite_trajectory.perigee_raising.continuous1d_a2c:main",
                                        "perigeeraising_continuous3d_a2c = phd_satellite_trajectory.perigee_raising.continuous3d_a2c:main",
                                        "perigeeraising_evaluate = phd_satellite_trajectory.perigee_raising.evaluate:main",
                                        "perigeeraising_plot_orbit = phd_satellite_trajectory.perigee_raising.plot_orbit:main",
                                        "perigeeraising_plot_plan = phd_satellite_trajectory.perigee_raising.plot_plan:main",
                                        "perigeeraising_plot_real = phd_satellite_trajectory.perigee_raising.plot_real:main",
                                        "perigeeraising_plot_training = phd_satellite_trajectory.perigee_raising.plot_training:main"],
                    "gui_scripts": [],
                    },
      description='Scripts for training agents that solve the environments for spacecraft trajectory optimization',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/zampanteymedio/phd-satellite-trajectory',
      keywords='gym openai env environment ai reinforcement learning spacecraft satellite trajectory optimization',
      license='Apache License 2.0',
      author='Carlos M. Casas Cuadrado',
      author_email='carlos.marce@gmail.com',
      install_requires=['gym>=0.17.3',
                        'gym-satellite-trajectory>=0.2+snapshot',
                        'matplotlib>=3.3.3',
                        'numpy>=1.19.4',
                        'setuptools>=51.0.0',
                        # TODO: Move stable-baselines3 to a stable release
                        'stable-baselines3>=0.11.0a2',
                        'torch>=1.7.0',
                        ],
      extras_require={'test': ['codecov>=2.1.10',
                               'coverage>=5.3',
                               'pytest>=6.1.2',
                               'pytest-cov>=2.10.0',
                               'pytest-ordering>=0.6',
                               ],
                      },
      )
