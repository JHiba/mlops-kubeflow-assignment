import kfp
from kfp import dsl

load_data_op = kfp.components.load_component_from_file('components/load_data.yaml')
preprocess_data_op = kfp.components.load_component_from_file('components/preprocess_data.yaml')
train_model_op = kfp.components.load_component_from_file('components/train_model.yaml')
evaluate_model_op = kfp.components.load_component_from_file('components/evaluate_model.yaml')

@dsl.pipeline(
    name='Offline Pipeline',
    description='Runs real code using local custom image.'
)
def mlops_pipeline():
    # Step 1: Generate Data
    load_task = load_data_op()
    load_task.container.set_image_pull_policy('Never') 
    
    # Step 2: Preprocess
    preprocess_task = preprocess_data_op(input_csv=load_task.outputs['output_csv'])
    preprocess_task.container.set_image_pull_policy('Never')
    
    # Step 3: Train
    train_task = train_model_op(train_data=preprocess_task.outputs['train_data'])
    train_task.container.set_image_pull_policy('Never')
    
    # Step 4: Evaluate
    evaluate_task = evaluate_model_op(
        test_data=preprocess_task.outputs['test_data'],
        model_input=train_task.outputs['model_output']
    )
    evaluate_task.container.set_image_pull_policy('Never')

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(mlops_pipeline, 'pipeline.yaml')
    print("Pipeline compiled successfully.")