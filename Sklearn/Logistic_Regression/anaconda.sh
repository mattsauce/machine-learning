# Go to home directory
cd ~

# You can change what anaconda version you want at 
# https://repo.continuum.io/archive/
curl -Ok https://repo.continuum.io/archive/Anaconda3-4.1.1-MacOSX-x86_64.sh
bash Anaconda3-4.1.1-MacOSX-x86_64.sh -b -p ~/anaconda
rm Anaconda3-4.1.1-MacOSX-x86_64.sh
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bash_profile 

# Refresh basically
source .bash_profile

conda update conda
