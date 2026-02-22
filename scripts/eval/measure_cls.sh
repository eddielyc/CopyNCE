echo "Calculate metrics from cls view."

python3 -m core.run.eval.measure_cls \
  --config-file "core/configs/eval/descriptor/vits_lin.yaml" \
  --output-dir "outputs" \
  --result-file "scores.json"
