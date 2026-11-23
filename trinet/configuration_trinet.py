"""TriNet model configuration for Hugging Face Transformers."""

from transformers import PretrainedConfig


class TriNetConfig(PretrainedConfig):
    """
    Configuration class for the 6-archetype heterogeneous TriNet architecture:
      1. TriChannelUnit        (associative working memory: W_base + dissolving F_fast)
      2. PyramidalUnit         (basal-vs-apical prediction mismatch & surprise)
      3. ShuntingInterneuron   (divisive Blue gain normalization)
      4. ModulatorNucleus      (slow tonic Yellow + sparse dead-zone write/retain gates)
      5. ReceptorBroadcastUnit (Michaelis-Menten 3-color chromaticity routing bus)
      6. DelayLineUnit         (K-tap causal shift register for local syntax/timing)
    """

    model_type = "trinet"
    attribute_map = {"num_hidden_layers": "n_layers"}

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        d_key: int = 64,
        d_val: int = 64,
        n_layers: int = 4,
        delay_lags: int = 4,
        n_receptors: int = 8,
        max_fast_weight: float = 6.0,
        max_write_rate: float = 1.5,
        min_retention: float = 0.5,
        residual_error: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_key = int(d_key)
        self.d_val = int(d_val)
        self.n_layers = int(n_layers)
        self.delay_lags = int(delay_lags)
        self.n_receptors = int(n_receptors)
        self.max_fast_weight = float(max_fast_weight)
        self.max_write_rate = float(max_write_rate)
        self.min_retention = float(min_retention)
        # Change 7 (provenance/MANIFEST.json): predict against W_base + F_fast.
        self.residual_error = bool(residual_error)
        self.initializer_range = float(initializer_range)
        self.use_cache = bool(use_cache)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


TriNetConfig.register_for_auto_class()
