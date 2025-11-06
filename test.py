import mlflow

x,y=75,10
z=x+y
#tracking the experiment with the mlflow

with mlflow.start_run():
    mlflow.log_param("x",x)
    mlflow.log_param("y",y)
    mlflow.log_metric("z",z)

print(z)
print("sunny")