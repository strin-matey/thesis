FROM ufoym/deepo:all-py36-jupyter
RUN pip install tabulate mpld3 jupyterthemes

RUN apt-get update 
RUN apt-get install -y \
    pandoc \
	texlive-xetex 

RUN jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T
