## Tech_Project
Implementing Docker with a Ubuntu system to briefly test performance of pandas, polars, and pyspark

## Instructions to obtain performance results
1) start up Docker Desktop
2) build docker image from the provided Dockerfile
  - docker image build -t pandas_test .
3) run the Docker container and mount it to your local directory (this for powershell)
  - docker run -it --rm -v ${PWD}:/app pandas_test
4) Once inside the container you can use linux to run the script
  - python3 pandas_performance_testing.py
5) Performance results are then found in your mounted directory entitled 'performance_results.csv'

