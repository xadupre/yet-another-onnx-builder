# Run only the examples inside the 'plotting' folder
python -m sphinx -D "sphinx_gallery_conf.filename_pattern=/tensorflow/" docs dist/html -j auto
python -m sphinx docs dist/html -j auto
