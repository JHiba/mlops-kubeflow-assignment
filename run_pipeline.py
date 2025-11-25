# import kfp

# # 1. Connect to the API (The Brain)
# # We will port-forward the API server to 8888
# client = kfp.Client(host='http://localhost:8888')

# # 2. Load the compiled pipeline file
# pipeline_filename = 'pipeline.yaml'

# # 3. Submit the Run
# print(f" Submitting {pipeline_filename} to Minikube...")

# try:
#     run_result = client.create_run_from_pipeline_package(
#         pipeline_file=pipeline_filename,
#         arguments={},  # No arguments needed for your simple pipeline
#         run_name='Task3_Manual_Run',
#         experiment_name='MLOps_Assignment'
#     )
    
#     print(f"✅ SUCCESS! Run submitted. Run ID: {run_result.run_id}")
#     print(" Waiting for the pipeline to finish execution...")
    
#     # 4. Wait for result
#     run_detail = client.wait_for_run_completion(run_result.run_id, timeout=600)
    
#     if run_detail.run.status == 'Succeeded':
#         print(" PIPELINE FINISHED SUCCESSFULLY!")
#     else:
#         print(f" Pipeline finished with status: {run_detail.run.status}")

# except Exception as e:
#     print(f" Connection Failed. Make sure port-forward is running! Error: {e}")


import kfp

API_HOST = "http://localhost:8888"    # ml-pipeline (API)
UI_HOST = "http://localhost:2746"     # ml-pipeline-ui (UI) if it ever runs

client = kfp.Client(host=API_HOST)
pipeline_filename = "pipeline.yaml"

print(f"🚀 Submitting {pipeline_filename} to Minikube...")

try:
    run_result = client.create_run_from_pipeline_package(
        pipeline_file=pipeline_filename,
        arguments={},  # No arguments needed
        run_name="Task3_Manual_Run",
        experiment_name="MLOps_Assignment",
    )

    run_id = run_result.run_id
    print(f"✅ SUCCESS! Run submitted. Run ID: {run_id}")

    # Build and print the UI URL
    run_url = f"{UI_HOST}/#/runs/details/{run_id}"
    print("\n🔗 If ml-pipeline-ui ever runs on 8080, this would be the run URL:")
    print(run_url)

    print("\n⏳ Waiting for the pipeline to finish execution...")
    run_detail = client.wait_for_run_completion(run_id, timeout=600)

    status = run_detail.run.status
    print(f"\n📌 Final run status: {status}")

    if status == "Succeeded":
        print("🎉 PIPELINE FINISHED SUCCESSFULLY!")
    else:
        print("⚠️ Pipeline did not succeed. Check kubeflow pods & logs.")

except Exception as e:
    print(f"❌ Connection Failed. Make sure port-forward is running! Error: {e}")
