for file in mistral.sh zephyr.sh mistral-chat.sh llama.sh llama-chat.sh
do
	sbatch $file
done

