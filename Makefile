.PHONY: faq history server test 

faq:
	python -m helpers.faq_manager

history:
	python -m helpers.history_manager

server:
	uvicorn app.main:app --reload

test:
	python -m helpers.test_script
