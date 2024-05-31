run_api:
	uvicorn startup_success_predictor.api.fast:app --reload

reinstall_package:
	@pip uninstall -y startup_success_predictor || :
	@pip install -e .


run_preprocess_train:
	python -c 'from startup_success_predictor.interface.main import preprocess_train; preprocess_train()'

run_pred:
	python -c 'from startup_success_predictor.interface.main import pred; pred()'

run_evaluate:
	python -c 'from startup_success_predictor.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate
