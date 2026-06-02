.PHONY: faq history server test audio

# Force UTF-8 stdout/stderr so emoji in script output don't crash on a
# default Windows console (cp1252 raises UnicodeEncodeError otherwise).
export PYTHONIOENCODING := utf-8

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