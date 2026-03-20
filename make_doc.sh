# Run only the examples inside the 'plotting' folder
@echo "--"
@echo "-- Run on 1 thread for tensorflow gallery"
@echo "--"
python -m sphinx -D "sphinx_gallery_conf.filename_pattern=/tensorflow/" docs dist/html -j 1
@echo "--"
@echo "-- Builds the documentation parallelized"
@echo "--"
python -m sphinx docs dist/html -j auto
