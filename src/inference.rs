use llm::{
    InferenceParameters, InferenceSession, LoadProgress,
    ModelKVMemoryType, TokenBias, InferenceSnapshot, ModelParameters,
    Model,
};
use rand::thread_rng;
use std::{convert::Infallible, io::BufReader};

use crate::cli_args::CLI_ARGS;
use flume::{unbounded, Receiver, Sender};

#[derive(Debug)]
pub struct InferenceRequest {
    /// The channel to send the tokens to.
    pub tx_tokens: Sender<Result<String, hyper::Error>>,

    pub num_predict: Option<usize>,
    pub prompt: String,
    pub n_batch: Option<usize>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub temp: Option<f32>,
    pub cache: u64,
}

pub fn initialize_model_and_handle_inferences() -> Sender<InferenceRequest> {
    // Create a channel for InferenceRequests and spawn a thread to handle them
    log::info!("ready");

    let (tx, rx) = unbounded();

    std::thread::spawn(move || {
        let args = &*CLI_ARGS;

        let mut inference_session_manager = InferenceSessionManager::new();

        let rx: Receiver<InferenceRequest> = rx;
        loop {
            if let Ok(inference_request) = rx.try_recv() {
                let mut session = inference_session_manager.get_session();
                let inference_params = InferenceParameters {
                    n_threads: args.num_threads,
                    n_batch: inference_request.n_batch.unwrap_or(args.batch_size),
                    top_k: inference_request.top_k.unwrap_or(args.top_k),
                    top_p: inference_request.top_p.unwrap_or(args.top_p),
                    repeat_penalty: inference_request
                        .repeat_penalty
                        .unwrap_or(args.repeat_penalty),
                    temperature: inference_request.temp.unwrap_or(args.temp),
                    bias_tokens: TokenBias::default(),
                    repetition_penalty_last_n: args.repeat_last_n,
                };
                let mut rng = thread_rng();
                
                // Run inference
                let model = &(inference_session_manager.model);
                let response = session.infer::<Infallible>(
                    model,
                    &mut rng,
                    &llm::InferenceRequest {
                        prompt: &inference_request.prompt,
                        parameters: Some(&inference_params),
                        play_back_previous_tokens: false,
                        maximum_token_count: None,
                    },
                    &mut llm::OutputRequest { all_logits: None, embeddings: None },
                    {
                        let tx_tokens = inference_request.tx_tokens.clone();
                        move |t| {
                            let text = t.to_string();
                            match tx_tokens.send(Ok(text)) {
                                Ok(_) => log::debug!("Sent token {} to receiver.", t),
                                Err(_) => log::warn!("Could not send token to receiver."),
                            }

                            Ok(())
                        }
                    },
                );
                if response.is_err() {
                    log::warn!("Failed to run.");
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });

    tx
}

/// `InferenceSessionManager` is a way to create new sessions for a model and vocabulary.
/// In the future, it can also manage how many sessions are created and manage creating sessions
/// between threads.
struct InferenceSessionManager {
    model: llm::models::Llama,
}

fn handle_progress(progress: LoadProgress) {
    match progress {
        LoadProgress::HyperparametersLoaded => {
            log::debug!("Loaded HyperParams")
        }
        LoadProgress::ContextSize { bytes } => log::info!(
            "ggml ctx size = {:.2} MB",
            bytes as f64 / (1024.0 * 1024.0)
        ),
        LoadProgress::TensorLoaded {
            current_tensor,
            tensor_count,
        } => log::info!(
            "Loading model part {}/{}",
            current_tensor,
            tensor_count,
        ),
        LoadProgress::Loaded {
            tensor_count,
            file_size,
        } => {
            log::info!("Loading complete");
            log::info!(
                "Model size = {:.2} MB / num tensors = {}",
                file_size as f64 / 1024.0 / 1024.0,
                tensor_count
            );
        }
    }
}

impl InferenceSessionManager {
    fn new() -> Self {
        // TODO It's not a great pattern to inject these arguments from CLI_ARGS.
        // If we ever wanted to support this struct in multiple places, please move the `args`
        // variable into properties of this struct.
        let args = &*CLI_ARGS;
        
        // set parameters
        let params = ModelParameters::default();
        
        // Load model
        let model = llm::models::Llama::load(&args.model_path, params, handle_progress)
            .expect("Could not load model");
        
        Self { model }
    }
    
    fn get_snapshot(&mut self) -> Option<InferenceSnapshot> {
        use std::fs::File;
        use zstd::stream::read::Decoder;

        let args = &*CLI_ARGS;
        
        if let Some(restore_path) = &args.restore_prompt {
            let file = File::open(restore_path).expect("could not open file");
            let decoder = Decoder::new(BufReader::new(file)).expect("could not create decoder");
            let snapshot = bincode::deserialize_from(decoder).expect("could not deserialize session");
            Some(snapshot)
        } else {
            None
        }
    }

    fn get_session(&mut self) -> InferenceSession {
        // TODO It's not a great pattern to inject these arguments from CLI_ARGS.
        // If we ever wanted to support this struct in multiple places, please move the `args`
        // variable into properties of this struct.
        let args = &*CLI_ARGS;
        
        if let Some(snapshot) = self.get_snapshot() {
            InferenceSession::from_snapshot(snapshot, &self.model).expect("could not create session")
        } else {
            let inference_session_params = {
                let mem_typ = if args.float16 {
                    ModelKVMemoryType::Float16
                } else {
                    ModelKVMemoryType::Float32
                };
                llm::InferenceSessionConfig {
                    memory_k_type: mem_typ,
                    memory_v_type: mem_typ,
                }
            };
            self.model.start_session(inference_session_params)
        }    
    }
}
