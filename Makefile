.PHONY: faq history server test audio

faq:
	python -m helpers.faq_manager

history:
	python -m helpers.history_manager

server:
	uvicorn app.main:app --reload

test:
	python -m helpers.test_script
audio: 
	python -m helpers.generate_audio_from_text