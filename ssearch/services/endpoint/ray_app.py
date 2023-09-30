# from ray import serve
# from fastapi import FastAPI

# app = FastAPI()

# @serve.deployment(
#     # ray_actor_options={"num_cpus": 12, "num_gpus": 0},
#     autoscaling_config={"min_replicas": 0, "max_replicas": 2},
#     max_concurrent_queries=100,
# )
# @serve.ingress(app)
# class SearchEndpoint:
#     def __init__(self):
#         pass

#     @app.get("/search")
#     async def search(self, t