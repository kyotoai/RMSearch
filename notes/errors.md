# Errors Note

## Oct 3

In search_tag function in "Reward Model Gets TopN-Relevant Sentences and LLM Judges Which Content Is More Relevant to A Question" in train_en.ipynb
```
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [dump_input.py:69] Dumping input data for V1 LLM engine (v0.10.1) with config: model='/workspace/llama3b-rm-converted-model', speculative_config=None, tokenizer='/workspace/llama3b-rm-converted-model', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=10000, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/workspace/llama3b-rm-converted-model, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=False, pooler_config=PoolerConfig(pooling_type='LAST', normalize=None, dimensions=None, activation=None, softmax=None, step_tag_id=None, returned_token_ids=None, enable_chunked_processing=None, max_embed_len=None), compilation_config={"level":3,"debug_dump_path":"","cache_dir":"/root/.cache/vllm/torch_compile_cache/6dec90b6ff","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":128,"local_cache_dir":"/root/.cache/vllm/torch_compile_cache/6dec90b6ff/rank_0_0/backbone"}, 
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [dump_input.py:76] Dumping scheduler output for model execution: SchedulerOutput(scheduled_new_reqs=[NewRequestData(req_id=4,prompt_token_ids_len=3699,mm_kwargs=[],mm_hashes=[],mm_positions=[],sampling_params=None,block_ids=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988],),num_computed_tokens=688,lora_request=None), NewRequestData(req_id=5,prompt_token_ids_len=3699,mm_kwargs=[],mm_hashes=[],mm_positions=[],sampling_params=None,block_ids=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177],),num_computed_tokens=688,lora_request=None), NewRequestData(req_id=6,prompt_token_ids_len=3698,mm_kwargs=[],mm_hashes=[],mm_positions=[],sampling_params=None,block_ids=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261],),num_computed_tokens=688,lora_request=None)], scheduled_cached_reqs=CachedRequestData(req_ids=['3'], resumed_from_preemption=[false], new_token_ids=[], new_block_ids=[[[747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799]]], num_computed_tokens=[2859]), num_scheduled_tokens={3: 840, 6: 1330, 4: 3011, 5: 3011}, total_num_scheduled_tokens=8192, scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={}, num_common_prefix_blocks=[43], finished_req_ids=['2', '1'], free_encoder_input_ids=[], structured_output_request_ids={}, grammar_bitmask=null, kv_connector_metadata=null)
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702] EngineCore encountered a fatal error.
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702] Traceback (most recent call last):
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 693, in run_engine_core
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     engine_core.run_busy_loop()
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 720, in run_busy_loop
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     self._process_engine_step()
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 745, in _process_engine_step
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     outputs, model_executed = self.step_fn()
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]                               ^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 288, in step
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     model_output = self.execute_model_with_error_logging(
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 274, in execute_model_with_error_logging
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     raise err
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 265, in execute_model_with_error_logging
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     return model_fn(scheduler_output)
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]            ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 87, in execute_model
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     output = self.collective_rpc("execute_model",
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     answer = run_method(self.driver_worker, method, args, kwargs)
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 3007, in run_method
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     return func(*args, **kwargs)
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     return func(*args, **kwargs)
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 362, in execute_model
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     output = self.model_runner.execute_model(scheduler_output,
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     return func(*args, **kwargs)
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1522, in execute_model
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     max_query_len) = (self._prepare_inputs(scheduler_output))
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 712, in _prepare_inputs
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]     tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702]               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(EngineCore_0 pid=2159) ERROR 10-03 10:28:01 [core.py:702] KeyError: None
(EngineCore_0 pid=2159) Process EngineCore_0:
(EngineCore_0 pid=2159) Traceback (most recent call last):
(EngineCore_0 pid=2159)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore_0 pid=2159)     self.run()
(EngineCore_0 pid=2159)   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore_0 pid=2159)     self._target(*self._args, **self._kwargs)
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 704, in run_engine_core
(EngineCore_0 pid=2159)     raise e
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 693, in run_engine_core
(EngineCore_0 pid=2159)     engine_core.run_busy_loop()
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 720, in run_busy_loop
(EngineCore_0 pid=2159)     self._process_engine_step()
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 745, in _process_engine_step
(EngineCore_0 pid=2159)     outputs, model_executed = self.step_fn()
(EngineCore_0 pid=2159)                               ^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 288, in step
(EngineCore_0 pid=2159)     model_output = self.execute_model_with_error_logging(
(EngineCore_0 pid=2159)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 274, in execute_model_with_error_logging
(EngineCore_0 pid=2159)     raise err
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 265, in execute_model_with_error_logging
(EngineCore_0 pid=2159)     return model_fn(scheduler_output)
(EngineCore_0 pid=2159)            ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 87, in execute_model
(EngineCore_0 pid=2159)     output = self.collective_rpc("execute_model",
(EngineCore_0 pid=2159)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
(EngineCore_0 pid=2159)     answer = run_method(self.driver_worker, method, args, kwargs)
(EngineCore_0 pid=2159)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 3007, in run_method
(EngineCore_0 pid=2159)     return func(*args, **kwargs)
(EngineCore_0 pid=2159)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=2159)     return func(*args, **kwargs)
(EngineCore_0 pid=2159)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 362, in execute_model
(EngineCore_0 pid=2159)     output = self.model_runner.execute_model(scheduler_output,
(EngineCore_0 pid=2159)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=2159)     return func(*args, **kwargs)
(EngineCore_0 pid=2159)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1522, in execute_model
(EngineCore_0 pid=2159)     max_query_len) = (self._prepare_inputs(scheduler_output))
(EngineCore_0 pid=2159)                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=2159)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 712, in _prepare_inputs
(EngineCore_0 pid=2159)     tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
(EngineCore_0 pid=2159)               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(EngineCore_0 pid=2159) KeyError: None
```
-> maybe batch_size problem or input length problem




## Oct 4

Inside vllm_reward2.py
```
File /workspace/RMS_exp/vllm_reward2.py:363, in LLMWorker.encode(self, prompts, pooling_params, batch_size, timeout_s)
    360     continue
    362 if "error" in payload:
--> 363     raise RuntimeError(f"Worker error in batch {item_idx}: {payload['error']}")
    365 # normal batch result
    366 chunk_results[item_idx] = payload["outputs"]

RuntimeError: Worker error in batch 0: EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.

```
-> restart the pod




In search_tag function in "Reward Model Gets TopN-Relevant Sentences and LLM Judges Which Content Is More Relevant to A Question" in train_en.ipynb

```
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596] WorkerProc hit an exception.
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596] Traceback (most recent call last):
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 591, in worker_busy_loop
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 362, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = self.model_runner.execute_model(scheduler_output,
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1522, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     max_query_len) = (self._prepare_inputs(scheduler_output))
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 712, in _prepare_inputs
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596] KeyError: None
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596] Traceback (most recent call last):
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 591, in worker_busy_loop
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 362, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = self.model_runner.execute_model(scheduler_output,
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1522, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     max_query_len) = (self._prepare_inputs(scheduler_output))
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 712, in _prepare_inputs
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596]               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596] KeyError: None
(EngineCore_0 pid=715) (VllmWorker TP0 pid=720) ERROR 10-04 05:07:25 [multiproc_executor.py:596] 
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596] WorkerProc hit an exception.
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596] Traceback (most recent call last):
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 591, in worker_busy_loop
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 362, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = self.model_runner.execute_model(scheduler_output,
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1522, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     max_query_len) = (self._prepare_inputs(scheduler_output))
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 712, in _prepare_inputs
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596] KeyError: None
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596] Traceback (most recent call last):
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 591, in worker_busy_loop
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 362, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     output = self.model_runner.execute_model(scheduler_output,
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     return func(*args, **kwargs)
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1522, in execute_model
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     max_query_len) = (self._prepare_inputs(scheduler_output))
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 712, in _prepare_inputs
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]     tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596]               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596] KeyError: None
(EngineCore_0 pid=715) (VllmWorker TP1 pid=722) ERROR 10-04 05:07:25 [multiproc_executor.py:596] 
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [dump_input.py:69] Dumping input data for V1 LLM engine (v0.10.1) with config: model='/workspace/llama3b-rm-converted-model', speculative_config=None, tokenizer='/workspace/llama3b-rm-converted-model', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4000, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/workspace/llama3b-rm-converted-model, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=False, pooler_config=PoolerConfig(pooling_type='LAST', normalize=None, dimensions=None, activation=None, softmax=None, step_tag_id=None, returned_token_ids=None, enable_chunked_processing=None, max_embed_len=None), compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":128,"local_cache_dir":null}, 
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [dump_input.py:76] Dumping scheduler output for model execution: SchedulerOutput(scheduled_new_reqs=[NewRequestData(req_id=4,prompt_token_ids_len=3699,mm_kwargs=[],mm_hashes=[],mm_positions=[],sampling_params=None,block_ids=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988],),num_computed_tokens=688,lora_request=None), NewRequestData(req_id=5,prompt_token_ids_len=3699,mm_kwargs=[],mm_hashes=[],mm_positions=[],sampling_params=None,block_ids=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177],),num_computed_tokens=688,lora_request=None), NewRequestData(req_id=6,prompt_token_ids_len=3698,mm_kwargs=[],mm_hashes=[],mm_positions=[],sampling_params=None,block_ids=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261],),num_computed_tokens=688,lora_request=None)], scheduled_cached_reqs=CachedRequestData(req_ids=['3'], resumed_from_preemption=[false], new_token_ids=[], new_block_ids=[[[747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799]]], num_computed_tokens=[2859]), num_scheduled_tokens={6: 1330, 4: 3011, 5: 3011, 3: 840}, total_num_scheduled_tokens=8192, scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={}, num_common_prefix_blocks=[43], finished_req_ids=['1', '2'], free_encoder_input_ids=[], structured_output_request_ids={}, grammar_bitmask=null, kv_connector_metadata=null)
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702] EngineCore encountered a fatal error.
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702] Traceback (most recent call last):
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 693, in run_engine_core
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     engine_core.run_busy_loop()
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 720, in run_busy_loop
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     self._process_engine_step()
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 745, in _process_engine_step
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     outputs, model_executed = self.step_fn()
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]                               ^^^^^^^^^^^^^^
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 288, in step
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     model_output = self.execute_model_with_error_logging(
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 274, in execute_model_with_error_logging
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     raise err
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 265, in execute_model_with_error_logging
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     return model_fn(scheduler_output)
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]            ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 173, in execute_model
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     (output, ) = self.collective_rpc(
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]                  ^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 243, in collective_rpc
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     result = get_response(w, dequeue_timeout)
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 230, in get_response
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702]     raise RuntimeError(
(EngineCore_0 pid=715) ERROR 10-04 05:07:25 [core.py:702] RuntimeError: Worker failed with error 'None', please check the stack trace above for the root cause
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[4], line 223
    210     key_dict.append({"query":df.iloc[i]["text"]})
    212 # key_dict: [{"query":"", **kwargs}, ...]
    213 # tag_dict: [{"tag":"", "children":[{"tag":""}, ...]}, ...]
    214 
   (...)    220 # query2tag_ids : [{"tag_ids":[[0,3,...], [2,1,...]]}, ...]   # (num_queries, "tag_ids":(k_tag, depth))
    221 # tag2query : [{"tag":"", "key_ids":[0,2, ...], "children":[{"key_ids":[2, ...]}, {"key_ids":[0, ...]}]}]
--> 223 query2tag_ids, tag2query = await search_tag(key_dict, tag_dict)
    225 with open(f"./data/{save_name}/query2tag_ids.json", "w") as f:
    226     json.dump(query2tag_ids, f)

Cell In[4], line 116, in search_tag(query_dict, tag_dict, k_tag)
    114 batch_size = total_requests // num_instances
    115 print(f"Graph Depth: {depth},  total_requests: {total_requests},  Batch size: {batch_size}")
--> 116 output = search(rm, requests, llm_template_func, topk = k_tag, batch_size=100, timeout_s=10000)
    118 with open(f"output{depth}.json", "w") as f:
    119     json.dump(output, f)

File /workspace/RMS_exp/vllm_reward2.py:445, in search(model, requests, llm_template, topk, **gen_kwargs)
    442 formatted_dataset = dataset1.map(format)
    443 df = formatted_dataset.to_pandas()
--> 445 rewards = model.encode(df["prompt"], **gen_kwargs)
    446 df["relevance"] = rewards.numpy()
    448 # Sort and pick topn per request

File /workspace/RMS_exp/vllm_reward2.py:363, in LLMWorker.encode(self, prompts, pooling_params, batch_size, timeout_s)
    360     continue
    362 if "error" in payload:
--> 363     raise RuntimeError(f"Worker error in batch {item_idx}: {payload['error']}")
    365 # normal batch result
    366 chunk_results[item_idx] = payload["outputs"]

RuntimeError: Worker error in batch 0: EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
(EngineCore_0 pid=715) Process EngineCore_0:
(EngineCore_0 pid=715) Traceback (most recent call last):
(EngineCore_0 pid=715)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore_0 pid=715)     self.run()
(EngineCore_0 pid=715)   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore_0 pid=715)     self._target(*self._args, **self._kwargs)
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 704, in run_engine_core
(EngineCore_0 pid=715)     raise e
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 693, in run_engine_core
(EngineCore_0 pid=715)     engine_core.run_busy_loop()
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 720, in run_busy_loop
(EngineCore_0 pid=715)     self._process_engine_step()
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 745, in _process_engine_step
(EngineCore_0 pid=715)     outputs, model_executed = self.step_fn()
(EngineCore_0 pid=715)                               ^^^^^^^^^^^^^^
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 288, in step
(EngineCore_0 pid=715)     model_output = self.execute_model_with_error_logging(
(EngineCore_0 pid=715)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 274, in execute_model_with_error_logging
(EngineCore_0 pid=715)     raise err
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 265, in execute_model_with_error_logging
(EngineCore_0 pid=715)     return model_fn(scheduler_output)
(EngineCore_0 pid=715)            ^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 173, in execute_model
(EngineCore_0 pid=715)     (output, ) = self.collective_rpc(
(EngineCore_0 pid=715)                  ^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 243, in collective_rpc
(EngineCore_0 pid=715)     result = get_response(w, dequeue_timeout)
(EngineCore_0 pid=715)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_0 pid=715)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 230, in get_response
(EngineCore_0 pid=715)     raise RuntimeError(
(EngineCore_0 pid=715) RuntimeError: Worker failed with error 'None', please check the stack trace above for the root cause
```
-> when input is quite long

