# Base image
FROM ubuntu:22.04

# Install Python and essential tools
RUN apt-get update && apt-get install -y python3 python3-pip openjdk-11-jdk

# Install Python libraries
RUN pip3 install pandas polars pyspark

# Set the working directory
WORKDIR /app

# Copy scripts and data into the container
COPY . .

