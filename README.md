# Rust HTTP Inference Server

This program is a simple HTTP server that uses the [llm](https://github.com/robvanvolt/llm) crate to run inference on a pre-trained language model and stream the results back to the client. It uses the [hyper](https://hyper.rs/) crate to handle HTTP requests and the [flume](https://docs.rs/flume/) crate to communicate between threads. It also uses the [clap](https://clap.rs/) crate to parse command-line arguments and the [serde](https://serde.rs/) crate to parse JSON data.

## Usage

To run this program, you need to have Rust installed on your system. You can install Rust using [rustup](https://rustup.rs/). You also need to have a pre-trained language model file in the [llm](https://github.com/robvanvolt/llm) format. You can download some example models from [here](https://github.com/robvanvolt/llm#pre-trained-models).

To compile and run this program, use the following command:

```bash
cargo run --release -- -m <model_path> [options]
```

where `<model_path>` is the path to the model file and `[options]` are optional arguments that you can specify. The available options are:

- `-P` or `--port`: The port to listen on. The default value is `8080`.
- `-f` or `--float16`: Use 16-bit floats for model memory key and value. Ignored when restoring from the cache. The default value is `false`.
- `-t` or `--num-threads`: Sets the number of threads to use. The default value is the number of physical cores on your system.
- `-n` or `--num-ctx-tokens`: Sets the size of the context (in tokens). Allows feeding longer prompts. Note that this affects memory. The default value is `2048`.
- `-b` or `--batch-size`: How many tokens from the prompt at a time to feed the network. Does not affect generation. This is the default value unless overridden by the request. The default value is `8`.
- `-r` or `--repeat-last-n`: Size of the 'last N' buffer that is used for the `repeat_penalty` option. In tokens. The default value is `64`.
- `-p` or `--repeat-penalty`: The penalty for repeating tokens. Higher values make the generation less likely to get into a loop, but may harm results when repetitive outputs are desired. This is the default value unless overridden by the request. The default value is `1.30`.
- `-T` or `--temp`: Temperature. This is the default value unless overridden by the request. The default value is `0.80`.
- `-k` or `--top-k`: Top-K: The top K words by score are kept during sampling. This is the default value unless overridden by the request. The default value is `40`.
- `-P` or `--top-p`: Top-p: The cummulative probability after which no more words are kept for sampling. This is the default value unless overridden by the request. The default value is `0.95`.
- `-R` or `--restore-prompt`: Restores a cached prompt at the given path, previously using --cache-prompt.

For example, to run this program with a model file named `model.llm`, using 16-bit floats and listening on port 8081, you can use this command:

```bash
cargo run --release -- -m model.llm -f -P 8081
```

## API

This program exposes a single endpoint: `/stream`. This endpoint accepts POST requests with a JSON body that specifies an inference request. The JSON body should have the following fields:

- `num_predict`: An optional field that specifies how many tokens to predict after the prompt. If not specified, it will use the default value of 32.
- `prompt`: A required field that specifies the prompt text to feed to the model.
- `n_batch`: An optional field that specifies how many tokens from the prompt at a time to feed the network. Does not affect generation. If not specified, it will use the default value specified by the command-line argument `-b` or `--batch-size`.
- `top_k`: An optional field that specifies the top K words by score to keep during sampling. If not specified, it will use the default value specified by the command-line argument `-k` or `--top-k`.
- `top_p`: An optional field that specifies the cummulative probability after which no more words are kept for sampling. If not specified, it will use the default value specified by the command-line argument `-P` or `--top-p`.
- `repeat_penalty`: An optional field that specifies the penalty for repeating tokens. Higher values make the generation less likely to get into a loop, but may harm results when repetitive outputs are desired. If not specified, it will use the default value specified by the command-line argument `-p` or `--repeat-penalty`.
- `temp`: An optional field that specifies the temperature. If not specified, it will use the default value specified by the command-line argument `-T` or `--temp`.
- `cache`: An optional field that specifies whether to cache the prompt and restore it later using the command-line argument `-R` or `--restore-prompt`. The value should be a positive integer that represents a unique identifier for the prompt. If not specified, it will use the default value of 0, which means no caching.

For example, a valid JSON body for an inference request could look like this:

```json
{
  "num_predict": 64,
  "prompt": "Once upon a time",
  "n_batch": 16,
  "top_k": 50,
  "top_p": 0.9,
  "repeat_penalty": 1.5,
  "temp": 0.7,
  "cache": 42
}
```

The response of this endpoint is a streaming body that sends tokens back to the client as they are generated by the model. The tokens are separated by newlines and encoded as UTF-8 strings. The client can read the tokens from the stream and display them as they wish.

For example, a possible response for the above request could look like this:

```text
Once upon a time
there was a princess named
```
