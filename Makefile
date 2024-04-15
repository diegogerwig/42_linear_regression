# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: marvin <marvin@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/03/09 20:16:45 by dgerwig-          #+#    #+#              #
#    Updated: 2024/04/15 22:50:04 by marvin           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

all: predict_after_train train predict_before_train precision test_lr

predict_after_train:
	@echo "\033[31mPredicting AFTER training...\033[0m"
	@python3 src/predict.py

train:
	@echo "\033[31mTraining...\033[0m\n"
	@python3 src/train.py

predict_before_train:
	@echo "\033[31mPredicting BEFORE training...\033[0m"
	@python3 src/predict.py
	@sleep 3

precision:
	@echo "\033[31mCalculating precision...\033[0m"
	@python3 src/precision.py

test_lr:
	@echo "\033[31mTesting diferent values for LEARNING RATE...\033[0m"
	@python3 src/test_learning_rate.py

clean:

fclean: clean
	@echo "\n🟡 Cleaning up...\n"
	@rm -rf **/__pycache__
	@rm -rf ./data/params.csv
	@rm -rf ./data/errors.csv
	@if [ ! -d "./plots_examples" ]; then \
		mkdir -p ./plots_examples; \
	fi
	@if [ -d "./plots" ]; then \
		cp -r ./plots/* ./plots_examples/; \
	fi
	@rm -rf ./plots
	
re:	fclean all
	
phony: all clean fclean re predict_after_train train predict_before_train precision test_lr
