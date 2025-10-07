{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;\f1\fnil\fcharset0 Menlo-Italic;}
{\colortbl;\red255\green255\blue255;\red66\green147\blue62;\red255\green255\blue255;\red42\green44\blue51;
\red147\green0\blue147;\red143\green144\blue150;\red50\green94\blue238;}
{\*\expandedcolortbl;;\cssrgb\c31373\c63137\c30980;\cssrgb\c100000\c100000\c100000;\cssrgb\c21961\c22745\c25882;
\cssrgb\c65098\c14902\c64314;\cssrgb\c62745\c63137\c65490;\cssrgb\c25098\c47059\c94902;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4 Models package for LLM Router Platform\
Model engines, wrappers, and inference abstractions\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2 """\cf4 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 from\cf4 \strokec4  .mistral_engine \cf5 \strokec5 import\cf4 \strokec4  (\
    
\f1\i \cf6 \strokec6 # Core classes
\f0\i0 \cf4 \strokec4 \
    ModelConfig \cf5 \strokec5 as\cf4 \strokec4  EngineModelConfig,\
    GenerationRequest,\
    GenerationResponse,\
    BaseModelEngine,\
    ModelManager,\
    \
    
\f1\i \cf6 \strokec6 # Engine implementations
\f0\i0 \cf4 \strokec4 \
    vLLMEngine,\
    HuggingFaceEngine,\
    RemoteModelEngine,\
    \
    
\f1\i \cf6 \strokec6 # Factory functions
\f0\i0 \cf4 \strokec4 \
    create_mistral_7b_config,\
    create_llama_7b_config,\
    create_codellama_config\
)\
\
\pard\pardeftab720\partightenfactor0

\f1\i \cf6 \strokec6 # Package metadata
\f0\i0 \cf4 \strokec4 \
__version__ \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "1.0.0"\cf4 \strokec4 \
__description__ \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "Model engines and inference abstractions"\cf4 \strokec4 \
\

\f1\i \cf6 \strokec6 # Export main components
\f0\i0 \cf4 \strokec4 \
__all__ \cf7 \strokec7 =\cf4 \strokec4  [\
    
\f1\i \cf6 \strokec6 # Core abstractions
\f0\i0 \cf4 \strokec4 \
    \cf2 \strokec2 'BaseModelEngine'\cf4 \strokec4 ,\
    \cf2 \strokec2 'ModelManager'\cf4 \strokec4 ,\
    \cf2 \strokec2 'EngineModelConfig'\cf4 \strokec4 ,\
    \cf2 \strokec2 'GenerationRequest'\cf4 \strokec4 , \
    \cf2 \strokec2 'GenerationResponse'\cf4 \strokec4 ,\
    \
    
\f1\i \cf6 \strokec6 # Engine implementations
\f0\i0 \cf4 \strokec4 \
    \cf2 \strokec2 'vLLMEngine'\cf4 \strokec4 ,\
    \cf2 \strokec2 'HuggingFaceEngine'\cf4 \strokec4 ,\
    \cf2 \strokec2 'RemoteModelEngine'\cf4 \strokec4 ,\
    \
    
\f1\i \cf6 \strokec6 # Factory functions
\f0\i0 \cf4 \strokec4 \
    \cf2 \strokec2 'create_mistral_7b_config'\cf4 \strokec4 ,\
    \cf2 \strokec2 'create_llama_7b_config'\cf4 \strokec4 ,\
    \cf2 \strokec2 'create_codellama_config'\cf4 \strokec4 \
]\
\

\f1\i \cf6 \strokec6 # Initialize logging for models package
\f0\i0 \cf4 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 import\cf4 \strokec4  logging\
logger \cf7 \strokec7 =\cf4 \strokec4  logging.getLogger(__name__)\
logger.info(\cf2 \strokec2 f"Models package v\cf4 \strokec4 \{__version__\}\cf2 \strokec2  initialized"\cf4 \strokec4 )}