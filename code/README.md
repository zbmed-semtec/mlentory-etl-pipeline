# Backend structure

In this backend folder you will find the code that is use to build and run the MLentory data extraction pipeline.

<img src="../docs/Readme_images/MLentory Backend TDD Diagrams-Main_component_interaction_Diagram_v2.png"/>
<p style=" text-align: center; font-size: 0.8em; color: #cccccc">MLentory Pipeline</p>



## Run the project

We use a docker based tools to run the project.

If you don't already have Docker, Docker Compose, and Nvdia container toolkit installed you can first check the [prerequisites section](#installing-prerequisites).

If you only want to check that everything runs you can execute the Test script that will build the whole architecture, you can execute *validate_tests.sh* script in the Test folder. Otherwise follow the instructions below:

Be sure to be inside the code folder.

1. Create the container network:
```console
docker network create mlentory_network
```

2. Enable the interaction of the containers with the docker engine:
```console
sudo chmod 666 /var/run/docker.sock
```


2. You need to build the images for the containers in the project, if you have a Nvidia gpu configured use the profile 'gpu' otherwise use the profile 'no_gpu':

```
docker-compose --profile gpu build
```
or
```
docker-compose --profile no_gpu build
```

3. Bring up the container architecture:

```
docker-compose --profile gpu up
```
or
```
docker-compose --profile no_gpu up
```

4. If you want to access any of the running containers:

```
docker exec -it <container_name> /bin/bash
```
For example, if you want to trigger a full dump of the Huggingface model data you can run the following command:
```console
# With GPU
docker exec hf_gpu python3 main.py
# Without GPU
docker exec hf_no_gpu python3 main.py
```


## Installing prerequisites
If you are in machine with a Unix based operating system you just need to install the Docker and Docker Compose services.

If you are in Windows we recommend installing the Windows subsystem for Linux (WSL 2) and install Ubuntu 20.04. The idea is to have a Linux machine inside Windows so that everything can run smoothly. Particularly when working with machine learning libraries using the Windows service for Docker can become troublesome.

### Setting up Docker on Linux

For Linux distribution like Ubuntu, Debian, CentOS, etc, we do the following:

1. Update your existing list of packages:
``` console
sudo apt update
```

2. Install a few prerequisite packages which let apt use packages over HTTPS:
``` console
sudo apt install apt-transport-https ca-certificates curl software-properties-common
```


3. Add the GPG key for the official Docker repository:
``` console
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```


4. Add the Docker repository to APT sources:
``` console
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
```


5. Update the package database with the Docker packages:
```
sudo apt update
```


7. Install Docker:
```
sudo apt install docker-ce
```


8. Verify the installation:
```
sudo docker run hello-world
```


### Manage Docker as Non-root User

If you don't want to write sudo before every command, do the following: 

1. Create the docker group if it does not exist:
```
sudo groupadd docker
```
2. Add your user to the docker group:
```
sudo usermod -aG docker ${USER}
```
3. Log out and log back in for changes to take effect.

4. Verify you can run Docker commands without sudo:
```
docker run hello-world
```

### Install Docker compose

1. Run this command to download the latest version of Docker Compose:
```
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```
2. Apply executable permissions to the binary:
```
sudo chmod +x /usr/local/bin/docker-compose
```
3. Verify the installation:
```
docker-compose --version
```

### Setup NVIDIA GPUs

* It is not necessary to have a gpu to run the Backend, but it will make the pipeline run faster.

* You can follow the guide at https://docs.nvidia.com/cuda/wsl-user-guide/index.html if you want to setup the NVDIA GPUs in your WSL.

* But in general you have to guarantee that you have the GPU drivers, the NVIDIA container toolkit, and you have CUDA toolkit install.

* If you are using Windows with WSL you have to install the GPU drivers in Windows, otherwise just install the drivers in your host OS. 
    * In Windows you can check the NVIDIA GPU drivers at: https://www.nvidia.com/Download/index.aspx
    * In Ubuntu you can check how to download the drivers at: https://ubuntu.com/server/docs/nvidia-drivers-installation
    * Remember to restart your system after installation.

If you don't have CUDA drivers installed to use your GPU for ML development you can follow the instructions here: 
https://developer.nvidia.com/cuda-downloads

### Update the default Docker DNS server
If you are using the WSL or a Linux distribution as your OS you need to configure the following in order for the private container network to resolve outside hostnames and interact correctly with the internet.

1. Install dnsmasq and resolvconf.
```
sudo apt update
sudo apt install dnsmasq resolvconf
```

2. Find your docker IP (in this case, 172.17.0.1):
```
root@host:~# ifconfig | grep -A2 docker0
docker0   Link encap:Ethernet  HWaddr 02:42:bb:b4:4a:50  
          inet addr:172.17.0.1  Bcast:0.0.0.0  Mask:255.255.0.0
```

3. Edit /etc/dnsmasq.conf and add these lines:
```
sudo nano /etc/dnsmasq.conf
```
```
interface=docker0
bind-interfaces
listen-address=172.17.0.1
```
5. Create/edit /etc/resolvconf/resolv.conf.d/tail (you can use vim or nano) and add this line, you have to change the line there with the IP of your default network interface eth0:
```
nameserver 8.8.8.8
```
6. Re-read the configuration files and regenerate /etc/resolv.conf.
```
sudo resolvconf -u
```
7. Restart your OS. If you are using WSL run the following in your windows terminal:
```
wsl.exe --shutdown
```
## Contribute to the project
If you are an external contributor from ZB MED you can fork the project and create a pull request.
Otherwise you can contact the ZB MED Semantic Technologies team to get access to the project directly.

Every time you want add a new feature or fix a bug you need to do the following:
1. Create a new branch with the name of the feature or bug you are fixing.
2. After making your changes you need to run the tests and make sure they pass.
    - You can run the tests by running the *validate_tests.sh* script in the Test folder.
3. Make sure to check the format of the code using black.
    - You can use black with the following command in the root folder of the project:
        ```
        black .
        ```
        You can run this command inside any instance/container that has access to the project files, and has python with black installed.
4. Commit your changes with a descriptive message and remember to use the [gitemoji format](https://gitmoji.dev/).
    - here an  example of a commit message:
        ```
        git commit -m "🐛 fix data leakage in the data extraction pipeline"
        ```
        By using the 🐛 symbol you can make clear you are solving a bug.

5. Push your changes to your branch.
6. Create a pull request to the main branch explaining the changes you made.
