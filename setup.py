from setuptools import setup,find_packages

setup(
    name="spectral_tool_box",
    version='0.0.1',    
    description="spectral_tool_box",
    url='https://github.com/felavila/Spectral_Tool_Box',
    author='Felipe Avila-Vera',
    author_email='felipe.avilav@postgrado.uv.cl',
    license='MIT License',
    packages=find_packages(include=["spectral_tool_box", "spectral_tool_box.*"]),
    install_requires=["astropy >=6.0", "numpy>=2.0","lmfit >=1.2.2","scipy >=1.11.4"
, "pandas>=2.2.0" ,"matplotlib>=3.8.2","parallelbar>=2.4", "pyspeckit>=1.0.3","csaps","ipywidgets"],
     classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
          'Programming Language :: Python :: 3.10',
    ],
)