import streamlit as st


class DockerInfo:

    def __init__(self):
        st.title("Docker Commands")

        # Define the main categories of Git commands
        categories = {
            "docker build -t docker_test .": [
                ("docker build",
                 "This is the command to build a Docker image"),
                ("-t docker_test",
                 "-t flag is used to specify a name and optionally a tag for the image. In this case, the image will be named docker_test"),
                (".",
                 ". indicates that the current directory should be used as the build context. This is where Docker looks for the Dockerfile and any files referenced in it"),
            ],
            "docker images": [
                ("docker images", "see all docker images on machine")
            ],
            "docker run -it --rm docker_test": [
                ("docker run", "runs docker container"),
                ("-it", "This combination of options stands for 'interactive' and 'tty.' It allows you to interact with the container's shell and provides a terminal interface."),
                ("--rm", "This option removes the container automatically after it exits. It's useful for temporary containers.")
            ],
            "Branch and merge": [
                ("git branch", "list your branches. a* will appear next to the currently active branch"),
                ("git branch [branch-name]", "create a new branch at the current commit"),
                ("git checkout", "switch to another branch and check it out into your working directory"),
                ("git merge [branch]", "merge the specified branch’s history into the current one"),
                ("git log", "show all commits in the current branch’s history"),
            ],

            "pip freeze > requirements.txt ": [
                ("pip freeze >", "saves all pip dependencies in file"),
                ("RUN pip install -r requirements.txt", "install requirements")
            ]
        }

        for category, commands in categories.items():
            st.header(category)
            st.divider()
            for command, description in commands:
                st.markdown(f'<p style="color: green;">{command}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: gray;">{description}</p>', unsafe_allow_html=True)
                st.write("")
                st.divider()
