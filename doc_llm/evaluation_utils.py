import json
from statistics import mean

from loguru import logger
from thefuzz import fuzz


def parse_json(prediction_str: str):
    prediction_str = prediction_str.replace("```json", "").replace("```", "")
    try:
        prediction = json.loads(prediction_str.strip())
    except json.decoder.JSONDecodeError:
        logger.error(f"Parsing error: \n {prediction_str}")

        prediction = {}

    return prediction


def fuzz_score_dicts(ground_truth: dict, prediction: dict):
    return mean(
        [
            fuzz.ratio(str(v), str(prediction.get(k, "")))
            for k, v in ground_truth.items()
        ]
    )


if __name__ == "__main__":
    mllm_output = """
        ```json
        {
          "company": "SAM SAM TRADING CO",
          "date": "742016-W",
          "address": "67, JLN Mewah 25/63 TWN SRI HUDA, 40400 SHAH ALAM.",
          "total": "RM 14.10"
        }
        ```
    """
    gt_output = """
        {
            "company": "SAM SAM TRADING CO",
            "date": "29-12-2017",
            "address": "67,JLN MEWAH 25/63 TMN SRI MUDA, 40400 SHAH ALAM.",
            "total": "14.10"
        }
    """

    _gt = parse_json(gt_output)
    _prediction = parse_json(mllm_output)

    print(fuzz_score_dicts(_gt, _prediction))
