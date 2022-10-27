# How to run

Fetch models
```yaml
git lfs fetch --all
```

Run triton and s3 instances
```yaml
docker-compose -f dev/docker-compose.yaml --env-file .env.dev up
```

The script shows a simplified image processing structure with triton
```yaml
python run.py --path `a path to your image`
```
