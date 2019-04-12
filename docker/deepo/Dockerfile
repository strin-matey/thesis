FROM ufoym/deepo:all-py36-jupyter
RUN pip install tabulate jupyterthemes
RUN pip install paramiko

RUN apt-get update 
RUN apt-get install -y \
    pandoc \
	texlive-xetex 

RUN jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T

RUN pip install seaborn

RUN python -m pip install --user "git+https://github.com/javadba/mpld3@display_fix"

