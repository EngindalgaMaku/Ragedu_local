"""
Model selection utility for RAG system.
Provides functionality to switch between different Ollama models dynamically.
"""

import ollama
import streamlit as st
from typing import Dict, List, Any, Optional
from ..config import get_available_models, set_generation_model, is_cloud_environment
from ..utils.logger import get_logger

class ModelSelector:
    """
    Handles model selection and validation for the RAG system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.ollama_client = None
        self._init_ollama_client()
    
    def _init_ollama_client(self):
        """Initialize Ollama client, skipping in cloud environments."""
        if is_cloud_environment():
            self.logger.info("Cloud environment detected, skipping Ollama client initialization.")
            self.ollama_client = None
            return
    
        try:
            self.ollama_client = ollama.Client(host=self.config.get("ollama_base_url"))
            self.ollama_client.list()
        except Exception as e:
            self.logger.warning(f"Could not connect to Ollama. Running without local models. Error: {e}")
            self.ollama_client = None
    
    def get_installed_models(self) -> List[str]:
        """Get list of currently installed models from Ollama."""
        if not self.ollama_client:
            return []
        
        try:
            models_response = self.ollama_client.list()
            installed_models = []
            
            # Ollama client returns models as a list in 'models' attribute
            for model in models_response.models:
                model_name = model.model  # Access the model name directly
                if model_name:
                    # Clean up model name (remove :latest if present)
                    if model_name.endswith(':latest'):
                        model_name = model_name[:-7]
                    installed_models.append(model_name)
            
            return installed_models
        except Exception as e:
            self.logger.error(f"Error getting installed models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is installed and available."""
        installed_models = self.get_installed_models()
        
        # Check various possible name formats
        model_variants = [
            model_name,
            f"{model_name}:latest",
            model_name.split(':')[0] if ':' in model_name else model_name
        ]
        
        for variant in model_variants:
            if variant in installed_models:
                return True
        
        return False
    
    def get_available_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models, both local and cloud."""
        # Start with local models from config
        available_models = get_available_models().copy()
        installed_models = self.get_installed_models()
        
        # Add any installed models not in config
        for installed_model in installed_models:
            if installed_model not in available_models:
                # Create basic info for unknown models
                available_models[installed_model] = {
                    'name': installed_model.replace(':', ' ').title(),
                    'description': f"Ollama'da yüklü model: {installed_model}",
                    'size': 'Bilinmiyor',
                    'performance': 'Bilinmiyor',
                    'language': 'Çok Dilli',
                    'provider': 'ollama'
                }
        
        # Add cloud models directly (they already have provider info)
        cloud_models = self.config.get('cloud_models', {})
        for model_key, model_info in cloud_models.items():
            available_models[model_key] = {
                **model_info,
                'installed': True,  # Cloud models are always "available"
                'status': "🌐 Ücretsiz",
                'status_color': "blue"
            }
        
        # Update status for local models
        for model_key, model_info in available_models.items():
            if model_info.get('provider') == 'ollama' or 'provider' not in model_info:
                # Check if this local model is installed
                model_info['installed'] = self.is_model_available(model_key)
                model_info['provider'] = 'ollama'
                
                # Add installation status
                if model_info['installed']:
                    model_info['status'] = "✅ Yüklü"
                    model_info['status_color'] = "green"
                else:
                    model_info['status'] = "⬇️ İndir"
                    model_info['status_color'] = "orange"
        
        return available_models
    
    def install_model(self, model_name: str) -> bool:
        """Install a model via Ollama."""
        if not self.ollama_client:
            self.logger.error("Cannot install model: Ollama client not available")
            return False
        
        try:
            self.logger.info(f"Starting installation of model: {model_name}")
            
            # Use Ollama CLI to pull the model
            import subprocess
            result = subprocess.run(
                ["ollama", "pull", model_name], 
                capture_output=True, 
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed model: {model_name}")
                return True
            else:
                self.logger.error(f"Failed to install model {model_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Model installation timed out: {model_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error installing model {model_name}: {e}")
            return False
    
    def test_model(self, model_name: str) -> bool:
        """Test if a model is working correctly."""
        if not self.ollama_client:
            return False
        
        try:
            # Simple test query
            response = self.ollama_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": "Merhaba, bu bir test mesajıdır."}],
                options={"num_predict": 20}
            )
            
            answer = response.get('message', {}).get('content', '')
            return len(answer.strip()) > 0
            
        except Exception as e:
            self.logger.error(f"Error testing model {model_name}: {e}")
            return False

def create_model_selector_ui():
    """Create Streamlit UI for model selection with provider choice."""
    st.subheader("🤖 AI Model Seçimi")
    
    # Initialize session state
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = 'groq'
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = st.session_state.get('ollama_generation_model', 'llama-3.1-8b-instant')
    
    # Provider Selection
    st.markdown("#### 🔧 Provider Seçimi")
    col_prov1, col_prov2, col_prov3 = st.columns(3)
    
    with col_prov1:
        if st.button("🏠 Ollama (Yerel)",
                     type="primary" if st.session_state.selected_provider == 'ollama' else "secondary",
                     use_container_width=True,
                     help="Yerel modeller - Bilgisayarınızda çalışır"):
            st.session_state.selected_provider = 'ollama'
            st.rerun()
    
    with col_prov2:
        if st.button("🌐 Groq (Cloud)",
                     type="primary" if st.session_state.selected_provider == 'groq' else "secondary",
                     use_container_width=True,
                     help="Cloud modeller - İnternet bağlantısı gerekir"):
            st.session_state.selected_provider = 'groq'
            st.rerun()
    
    with col_prov3:
        if st.button("🚀 OpenRouter",
                     type="primary" if st.session_state.selected_provider == 'openrouter' else "secondary",
                     use_container_width=True,
                     help="OpenRouter API - Premium AI modelleri"):
            st.session_state.selected_provider = 'openrouter'
            st.rerun()
    
    st.markdown("---")
    
    # Get available models with status - PROVIDER AWARE
    available_models = get_available_models_info(st.session_state.selected_provider)
    
    # Filter models by selected provider
    if st.session_state.selected_provider == 'ollama':
        filtered_models = {k: v for k, v in available_models.items()
                          if v.get('provider', 'ollama') == 'ollama'}
        st.info("🏠 **Yerel modeller** - Bilgisayarınızda çalışır, internet gerektirmez")
    elif st.session_state.selected_provider == 'openrouter':
        filtered_models = {k: v for k, v in available_models.items()
                          if v.get('provider') == 'openrouter'}
        st.info("🚀 **OpenRouter modelleri** - Premium AI modelleri, çeşitli sağlayıcılar")
    else:  # groq
        filtered_models = {k: v for k, v in available_models.items()
                          if v.get('provider') == 'groq'}
        st.info("🌐 **Groq modelleri** - Hızlı ve güçlü, internet bağlantısı gerekir")
    
    if not filtered_models:
        st.warning(f"⚠️ {st.session_state.selected_provider.title()} için model bulunamadı")
        return st.session_state.selected_model
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model selection dropdown for selected provider
        model_options = []
        model_labels = []
        
        # Sort models: installed first, then by name
        sorted_models = sorted(
            filtered_models.items(),
            key=lambda x: (not x[1].get('installed', False), x[1]['name'])
        )
        
        for model_key, model_info in sorted_models:
            model_options.append(model_key)
            status = model_info.get('status', '❓ Bilinmiyor')
            label = f"{model_info['name']} - {status}"
            model_labels.append(label)
        
        # Create mapping for display
        model_mapping = dict(zip(model_labels, model_options))
        
        # Current selection - if current model not in filtered, use first available
        current_model_key = st.session_state.selected_model
        current_label = None
        for label, key in model_mapping.items():
            if key == current_model_key:
                current_label = label
                break
        
        # If current model not in filtered provider, select first one
        if not current_label and model_labels:
            current_label = model_labels[0]
            current_model_key = model_mapping[current_label]
        
        selected_label = st.selectbox(
            f"🤖 {st.session_state.selected_provider.title()} Modeli:",
            options=model_labels,
            index=model_labels.index(current_label) if current_label else 0,
            help=f"Kullanılacak {st.session_state.selected_provider} modelini seçin"
        )
        
        selected_model = model_mapping[selected_label]
        
        # Update session state
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            set_generation_model(selected_model)
            st.success(f"✅ Model değiştirildi: {filtered_models[selected_model]['name']}")
    
    with col2:
        # Model actions
        selected_info = filtered_models[selected_model]
        
        if st.session_state.selected_provider == 'ollama':
            # Ollama model actions
            if selected_info.get('installed', False):
                st.success("✅ Yüklü")
                if st.button("🧪 Test Et", help="Modeli test et"):
                    with st.spinner("Model test ediliyor..."):
                        try:
                            # Initialize ModelSelector with proper config
                            from src.config import get_config
                            selector = ModelSelector(get_config())
                            if selector.test_model(selected_model):
                                st.success("✅ Model çalışıyor!")
                            else:
                                st.error("❌ Model test başarısız")
                        except Exception as e:
                            st.error(f"❌ Test hatası: {e}")
            else:
                st.warning("⬇️ Yüklü değil")
                if st.button(f"📥 {selected_model} İndir", help="Modeli indir"):
                    with st.spinner("Model indiriliyor... Bu işlem uzun sürebilir."):
                        try:
                            # Initialize ModelSelector with proper config
                            from src.config import get_config
                            selector = ModelSelector(get_config())
                            if selector.install_model(selected_model):
                                st.success("✅ Model başarıyla yüklendi!")
                                st.rerun()
                            else:
                                st.error("❌ Model yüklenemedi")
                        except Exception as e:
                            st.error(f"❌ İndirme hatası: {e}")
        elif st.session_state.selected_provider == 'openrouter':
            # OpenRouter cloud model - always available if API key is set
            st.success("🚀 OpenRouter Hazır")
            if st.button("🧪 Test Et", help="OpenRouter API bağlantısını test et"):
                with st.spinner("OpenRouter API test ediliyor..."):
                    try:
                        from ..utils.cloud_llm_client import CloudLLMClient
                        client = CloudLLMClient()
                        test_response = client.generate_response(
                            "Merhaba, bu bir test mesajıdır.",
                            selected_model,
                            'openrouter'
                        )
                        if "❌" not in test_response:
                            st.success("✅ OpenRouter API çalışıyor!")
                        else:
                            st.error(f"❌ Test başarısız: {test_response}")
                    except Exception as e:
                        st.error(f"❌ OpenRouter test hatası: {e}")
        else:
            # Groq cloud model - always available
            st.success("🌐 Cloud Hazır")
            if st.button("🧪 Test Et", help="Cloud bağlantısını test et"):
                with st.spinner("Groq API test ediliyor..."):
                    try:
                        from ..utils.cloud_llm_client import CloudLLMClient
                        client = CloudLLMClient()
                        test_response = client.generate_response(
                            "Merhaba, bu bir test mesajıdır.",
                            selected_model,
                            'groq'
                        )
                        if "❌" not in test_response:
                            st.success("✅ Groq API çalışıyor!")
                        else:
                            st.error(f"❌ Test başarısız: {test_response}")
                    except Exception as e:
                        st.error(f"❌ Cloud test hatası: {e}")
    
    # Model information
    with st.expander("📋 Model Bilgileri", expanded=False):
        selected_info = available_models[selected_model]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Boyut", selected_info.get('size', 'Bilinmiyor'))
        
        with col2:
            st.metric("Performans", selected_info.get('performance', 'Bilinmiyor'))
        
        with col3:
            st.metric("Dil Desteği", selected_info.get('language', 'Bilinmiyor'))
        
        st.info(f"**Açıklama:** {selected_info.get('description', 'Açıklama mevcut değil')}")
    
    return selected_model

def create_simple_model_selector_ui():
    """Öğrenci paneli için basitleştirilmiş model seçici - provider + model."""
    
    # Initialize session state
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = 'groq'
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = st.session_state.get('ollama_generation_model', 'llama-3.1-8b-instant')
    
    # Provider Selection - Compact
    st.markdown("#### 🔧 AI Provider")
    provider_col1, provider_col2, provider_col3 = st.columns(3)
    
    with provider_col1:
        if st.button("🏠 Yerel",
                     type="primary" if st.session_state.selected_provider == 'ollama' else "secondary",
                     use_container_width=True):
            st.session_state.selected_provider = 'ollama'
            st.rerun()
    
    with provider_col2:
        if st.button("🌐 Groq",
                     type="primary" if st.session_state.selected_provider == 'groq' else "secondary",
                     use_container_width=True):
            st.session_state.selected_provider = 'groq'
            st.rerun()
    
    with provider_col3:
        if st.button("🚀 OpenRouter",
                     type="primary" if st.session_state.selected_provider == 'openrouter' else "secondary",
                     use_container_width=True):
            st.session_state.selected_provider = 'openrouter'
            st.rerun()
    
    # Get available models with status - PROVIDER AWARE
    available_models = get_available_models_info(st.session_state.selected_provider)
    
    # Filter models by selected provider
    if st.session_state.selected_provider == 'ollama':
        # Yerel modeller: yüklü olanlar
        usable_models = {k: v for k, v in available_models.items()
                        if v.get('provider', 'ollama') == 'ollama' and v.get('installed', False)}
        provider_info = "🏠 Yerel modeller"
    elif st.session_state.selected_provider == 'openrouter':
        # OpenRouter modeller: her zaman kullanılabilir
        usable_models = {k: v for k, v in available_models.items()
                        if v.get('provider') == 'openrouter'}
        provider_info = "🚀 OpenRouter modeller"
    else:  # groq
        # Cloud modeller: her zaman kullanılabilir
        usable_models = {k: v for k, v in available_models.items()
                        if v.get('provider') == 'groq'}
        provider_info = "🌐 Groq modeller"
    
    if not usable_models:
        if st.session_state.selected_provider == 'ollama':
            st.info("ℹ️ Yerel model yükleniyor... Öğretmen panelinden kontrol edebilirsiniz.")
        elif st.session_state.selected_provider == 'openrouter':
            st.warning("⚠️ OpenRouter API anahtarı gerekli (.env dosyasında)")
        else:
            st.warning("⚠️ Groq API anahtarı gerekli (.env dosyasında)")
        return st.session_state.selected_model
    
    # Model selection
    model_options = list(usable_models.keys())
    model_labels = []
    
    for k, v in usable_models.items():
        provider_icon = "🏠" if v.get('provider', 'ollama') == 'ollama' else "🌐"
        model_labels.append(f"{provider_icon} {v['name']}")
    
    # Current selection - if current model not available in provider, use first
    current_model_key = st.session_state.selected_model
    current_index = 0
    if current_model_key in model_options:
        current_index = model_options.index(current_model_key)
    
    selected_index = st.selectbox(
        f"🤖 {provider_info}:",
        options=range(len(model_labels)),
        format_func=lambda x: model_labels[x],
        index=current_index,
        help="Hangi AI modeli kullanılsın?"
    )
    
    selected_model = model_options[selected_index]
    
    # Update session state - with provider sync
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Update provider based on selected model
        model_provider = usable_models[selected_model].get('provider', 'ollama')
        if model_provider == 'groq':
            st.session_state.selected_provider = 'groq'
        elif model_provider == 'openrouter':
            st.session_state.selected_provider = 'openrouter'
        else:
            st.session_state.selected_provider = 'ollama'
        set_generation_model(selected_model)
    
    return selected_model

def get_available_models_info(provider_filter: str = None) -> Dict[str, Dict[str, Any]]:
    """Standalone function to get available models info for UI components - PROVIDER AWARE."""
    from src import config as app_config
    import streamlit as st
    
    models_info = {}
    
    # Session state'den current provider'ı al
    current_provider = st.session_state.get('selected_provider', 'groq')
    if provider_filter:
        current_provider = provider_filter
    
    # 1. CLOUD_MODELS'den external API modellerini ekle (groq ve openrouter için)
    if current_provider in ['groq', 'openrouter', 'cloud']:
        for model_name, model_config in app_config.CLOUD_MODELS.items():
            model_provider = model_config.get("provider", "groq")
            # Sadece current provider ile eşleşenleri ekle
            if current_provider == model_provider or current_provider == 'cloud':
                models_info[model_name] = {
                    "type": "cloud",
                    "status": "🌐 External API" if model_provider == 'groq' else "🚀 Premium API",
                    "name": model_config.get("name", model_name),
                    "description": model_config.get("description", f"External API: {model_name}"),
                    "provider": model_provider,
                    "installed": True  # Cloud modeller her zaman kullanılabilir
                }
    
    # 2. Ollama modellerini sadece ollama provider seçiliyse kontrol et
    if current_provider == 'ollama':
        # AVAILABLE_MODELS'den yerel modelleri al
        for model_name, model_config in app_config.AVAILABLE_MODELS.items():
            is_cloud = "cloud" in model_name.lower() or model_config.get("provider") == "cloud"
            if not is_cloud:  # Sadece yerel modeller
                models_info[model_name] = {
                    "type": "ollama",
                    "status": "⬇️ Yüklenmemiş",
                    "name": model_config.get("name", model_name),
                    "description": model_config.get("description", f"Model: {model_name}"),
                    "size": model_config.get("size", "Bilinmiyor"),
                    "performance": model_config.get("performance", "Medium"),
                    "installed": False,
                    "provider": "ollama"
                }
        
        # Gerçekte yüklü Ollama modellerini kontrol et - SADECE OLLAMA SEÇILIYSE
        try:
            # Only try to connect to Ollama if we have Ollama models to check
            if any(model_info["type"] == "ollama" for model_info in models_info.values()):
                import ollama
                client = ollama.Client(host=app_config.OLLAMA_BASE_URL)
                models_response = client.list()
                
                installed_models = []
                for model in models_response.models:
                    model_name = model.model
                    if model_name.endswith(':latest'):
                        model_name = model_name[:-7]
                    installed_models.append(model_name)
                    
                    # Gerçekte yüklü modeli işaretle veya ekle
                    if model_name in models_info:
                        models_info[model_name]["status"] = "✅ Hazır"
                        models_info[model_name]["installed"] = True
                    else:
                        models_info[model_name] = {
                            "type": "ollama",
                            "status": "✅ Hazır",
                            "name": model_name.replace(':', ' ').title(),
                            "description": f"Yüklü Ollama modeli: {model_name}",
                            "provider": "ollama",
                            "installed": True
                        }
            
        except Exception as e:
            # Ollama çevrimdışıysa veya olmadığında yerel modelleri işaretle
            logger_msg = f"Ollama kontrol edilemedi (bu normal bir durum cloud provider kullanırken): {e}"
            try:
                import logging
                logging.getLogger(__name__).info(logger_msg)
            except:
                print(logger_msg)
                
            for model_name in models_info:
                if models_info[model_name]["type"] == "ollama":
                    if "timeout" in str(e).lower() or "connection" in str(e).lower():
                        models_info[model_name]["status"] = "⚠️ Ollama Çevrimdışı"
                    else:
                        models_info[model_name]["status"] = "⬇️ Yüklenmemiş"
                    models_info[model_name]["installed"] = False
    
    return models_info

def get_current_model():
    """Get the currently selected model."""
    return st.session_state.get('selected_model', 'mistral:7b')