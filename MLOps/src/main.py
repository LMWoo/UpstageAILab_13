import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import fire
from icecream import ic
import wandb
from dotenv import load_dotenv

from src.dataset.watch_log import get_datasets
from src.dataset.data_loader import SimpleDataLoader
from src.model.movie_predictor import MoviePredictor, model_save
from src.utils.utils import init_seed, auto_increment_run_suffix
from src.train.train import train
from src.evaluate.evaluate import evaluate
from src.utils.constant import Models


init_seed()
load_dotenv()


def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")


def get_latest_run(project_name):
    runs = get_runs(project_name)
    if not runs:
        return f"{project_name}-000"
    
    return runs[0].name


def run_train(model_name, batch_size=64, num_epochs=10):
    Models.validation(model_name)

    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = model_name.replace("_", "-") # movie_predictor -> movie-predictor
    run_name = get_latest_run(project_name)
    next_run_name = auto_increment_run_suffix(run_name)

    wandb.init(
        project=project_name,
        id=next_run_name,
        name=next_run_name,
        notes="content-based movie recommend model",
        tags=["content-based", "movie", "recommend"],
        config=locals(),
    )

    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = SimpleDataLoader(train_dataset.features, train_dataset.labels, batch_size=batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_dataset.features, val_dataset.labels, batch_size=batch_size, shuffle=False)
    test_loader = SimpleDataLoader(test_dataset.features, test_dataset.labels, batch_size=batch_size, shuffle=False)

    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes,
        "hidden_dim": 64,
    }
    # model = MoviePredictor(**model_params)
    model_class = Models[model_name.upper()].value  # Models -> MOVIE_PREDICTOR = MoviePredictor
    model = model_class(**model_params)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader)
        val_loss, _ = evaluate(model, val_loader)
        ic(f"{epoch + 1}/{num_epochs}")
        ic(train_loss)
        ic(val_loss)
        wandb.log({"Loss/Train": train_loss})
        wandb.log({"Loss/Valid": val_loss})
    
    test_loss, predictions = evaluate(model, test_loader)
    ic(test_loss)
    ic([train_dataset.decode_content_id(idx) for idx in predictions])
    
    model_save(
        model=model,
        model_params=model_params,
        epoch=num_epochs,
        loss=train_loss,
        scaler=train_dataset.scaler,
        label_encoder=train_dataset.label_encoder,
    )

    wandb.finish()


if __name__ == '__main__':  # python main.py
    fire.Fire({
        "train": run_train,  # python main.py train --model_name movie_predictor
    })
