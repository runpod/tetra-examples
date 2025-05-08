### Auto Environment Loading for ServerlessResource
## Overview
The `ServerlessResource` class automatically loads environment variables from your .env file, simplifying configuration management for serverless deployments.
Features

- ***Zero-configuration***: Environment variables load automatically when creating a ServerlessResource
- ***Override capability***: Custom environment variables can still be provided when needed
- ***Simple integration***: Works with existing .env file.


```python
# Create with auto-loaded environment variables
resource = ServerlessResource(
    name="inference-api",
    templateId="template123"
    # env automatically populated from .env file
)
```

or 

```python
# You can Override the env vars in config
resource = ServerlessResource(
    name="inference-api",
    templateId="template123"
    env={"CUSTOM_VAR": "custom_value"}
)
```
