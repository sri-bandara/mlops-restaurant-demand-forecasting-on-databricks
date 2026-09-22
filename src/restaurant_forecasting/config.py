"""Reads the project_config.yml file and turns it into a ProjectConfig object."""

from typing import Any #any is a generic type that can be used to represent any type
import yaml #for reading the project_config.yml file
from pydantic import BaseModel #Pydantic is a library for data validation and BaseModel is its main class for checking if the data matches the expected type


class ProjectConfig(BaseModel): #inherits from BaseModel, which means it will have all the methods and attributes of BaseModel
    """Holds and validates project configuration loaded from YAML."""

    catalog_name: str
    schema_name: str
    experiment_name_basic: str
    target: str
    num_features: list[str] #should be a list of strings
    cat_features: list[str]
    parameters: dict[str, Any] #should be a dictionary mapping strings to any type of value

    @classmethod #methods usually work on objects, but class methods can be called on the class itself. 
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Loads the project configuration from a YAML file and returns a ProjectConfig object.

        Args:
            cls: The class itself 
            config_path (str): Path to the project_config.yml file.
            env (str): Environment to load the configuration for. Defaults to "dev".    
        
        Returns:
            ProjectConfig: An instance of the ProjectConfig class with the loaded configuration.
        """

        with open(config_path) as f:
            config_dict = yaml.safe_load(f) #safe_load loads the content from f as a Python dictionary, which is then stored in config_dict

        #extracts the environment-specific configuration from the loaded dictionary
        env_config = config_dict[env]

        return cls( #creates an object of the ProjectConfig class and returns it
            catalog_name=env_config["catalog_name"],
            schema_name=env_config["schema_name"],
            experiment_name_basic=config_dict["experiment_name_basic"],
            target=config_dict["target"],
            num_features=config_dict["num_features"],
            cat_features=config_dict["cat_features"],
            parameters=config_dict["parameters"],
        )


class Tags(BaseModel):
    """Holds and validates the git SHA and the branch."""

    git_sha: str 
    branch: str