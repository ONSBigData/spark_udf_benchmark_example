#!/usr/bin/env bash

DOWNLOAD_DIRECTORY=/vagrant/resources/downloads
CHECKSUM_DIR=/vagrant/resources/checksums

MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
MINICONDA_CHECKSUM=${CHECKSUM_DIR}/miniconda.md5sum.txt

SPARK_URL=http://apache.mirrors.nublue.co.uk/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
SPARK_CHECKSUM=${CHECKSUM_DIR}/spark-2.3.1-bin-hadoop2.7.tgz.sha512.txt

# Prep the box
apt-get -y update
apt-get -y upgrade

# Install Java
apt-get install -y openjdk-8-jdk-headless

# Download Miniconda3
if [[ ! -e ${DOWNLOAD_DIRECTORY}/Miniconda3-latest-Linux-x86_64.sh ]]; then
    wget --no-verbose --directory-prefix=$DOWNLOAD_DIRECTORY $MINICONDA_URL
fi

# Verify the download
if md5sum --check $MINICONDA_CHECKSUM; then
    echo "checked md5sum for miniconda"
else
    echo "bad md5sum for miniconda, exiting script"
    exit 1
fi

# Install Miniconda
if [[ ! -e /home/vagrant/resources/checksums/miniconda ]]; then
    bash ${DOWNLOAD_DIRECTORY}/Miniconda3-latest-Linux-x86_64.sh -b -p /home/vagrant/miniconda
fi

sudo chown -R vagrant:vagrant /home/vagrant/miniconda


# Add some arguments to .bashrc
#TODO make this idempotent
echo 'source /vagrant/resources/.bashrc_extralines' >> /home/vagrant/.bashrc

apt-get -y install scala

# Download Spark
if [[ ! -e ${DOWNLOAD_DIRECTORY}/spark-2.3.1-bin-hadoop2.7.tgz ]]; then
    echo "Downloading Spark, this can take a while, and wget is set to --no-verbose"
    echo "If impatient check the file in $DOWNLOAD_DIRECTORY"
    wget --no-verbose --directory-prefix=$DOWNLOAD_DIRECTORY $SPARK_URL
fi

# Verify Spark download
if sha512sum --check $SPARK_CHECKSUM; then
    echo "checked spark sha512"
else
    echo "Bad sha512 for spark tgz, exiting script"
    exit 1
fi

# Unpack Spark
tar -xzf ${DOWNLOAD_DIRECTORY}/spark-2.3.1-bin-hadoop2.7.tgz -C /usr/local

if [[ ! -e /usr/local/spark ]]; then
    ln -s /usr/local/spark-2.3.1-bin-hadoop2.7 /usr/local/spark
fi

# Move any scripts to user home
cp /vagrant/scripts/*.sh /home/vagrant
chmod +x /home/vagrant/*.sh

# Set up miniconda to run here
. /home/vagrant/miniconda/etc/profile.d/conda.sh

conda activate

# Install Python dependencies
conda install pyspark pandas numpy nltk matplotlib seaborn --yes

python -m pip install jellyfish
