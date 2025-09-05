#!/usr/bin/env python3
"""
Interactive Async Strands Demo - Long-Running Data Processing

This example demonstrates realistic long-running background tasks with:
- 30-minute data processing simulation (configurable)
- Real-time progress tracking via result files
- User-configurable parameters (dataset size, processing type, etc.)
- Proper async task lifecycle management
- Agent remains fully interactive during processing

Key Features:
✅ Long-running background processing (30 minutes default)
✅ Real-time progress updates (every second to file)
✅ Multiple processing stages with realistic timing
✅ Interactive progress monitoring
✅ Proper task tracking with app.add_async_task() / app.complete_async_task()
✅ Agent stays responsive throughout
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from strands import Agent, tool

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Configure logging with INFO level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize app with interactive task control
app = BedrockAgentCoreApp(debug=True)

# Global task registry to track active tasks
active_tasks = {}


class DataProcessor:
    """Simulates realistic data processing with multiple stages."""

    PROCESSING_STAGES = [
        {"name": "data_loading", "weight": 0.10, "description": "Loading dataset"},
        {"name": "data_validation", "weight": 0.15, "description": "Validating data integrity"},
        {"name": "preprocessing", "weight": 0.25, "description": "Cleaning and preprocessing"},
        {"name": "feature_extraction", "weight": 0.30, "description": "Extracting features"},
        {"name": "analysis", "weight": 0.15, "description": "Running analysis"},
        {"name": "results_generation", "weight": 0.05, "description": "Generating results"},
    ]

    def __init__(
        self, task_id: int, dataset_size: str, processing_type: str, duration_minutes: int = 30, batch_size: int = 100
    ):
        self.task_id = task_id
        self.dataset_size = dataset_size
        self.processing_type = processing_type
        self.duration_minutes = duration_minutes
        self.batch_size = batch_size

        # Calculate total items based on dataset size
        size_multipliers = {"small": 500, "medium": 2000, "large": 5000, "huge": 10000}
        self.total_items = size_multipliers.get(dataset_size.lower(), 2000)

        self.start_time = datetime.now()
        self.result_file = f"data_processing_results_{task_id}.json"
        self.items_processed = 0
        self.current_stage_index = 0
        self.stage_start_time = time.time()

        # Calculate processing speed (items per second)
        total_seconds = duration_minutes * 60
        self.base_processing_speed = self.total_items / total_seconds

    def get_current_stage(self) -> Dict[str, Any]:
        """Get current processing stage info."""
        if self.current_stage_index < len(self.PROCESSING_STAGES):
            return self.PROCESSING_STAGES[self.current_stage_index]
        return {"name": "completed", "weight": 0, "description": "Processing completed"}

    def calculate_progress(self) -> Dict[str, Any]:
        """Calculate detailed progress information."""
        current_stage = self.get_current_stage()

        # Calculate overall progress based on completed stages + current stage progress
        completed_weight = sum(stage["weight"] for stage in self.PROCESSING_STAGES[: self.current_stage_index])

        # Current stage progress (0-1)
        stage_progress = min(
            1.0,
            (self.items_processed % (self.total_items // len(self.PROCESSING_STAGES)))
            / (self.total_items // len(self.PROCESSING_STAGES)),
        )

        current_stage_weight = current_stage["weight"] * stage_progress
        overall_progress = min(100.0, (completed_weight + current_stage_weight) * 100)

        # Calculate ETA
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        if overall_progress > 0:
            total_estimated_seconds = (elapsed_seconds / overall_progress) * 100
            remaining_seconds = max(0, total_estimated_seconds - elapsed_seconds)
            eta = datetime.now() + timedelta(seconds=remaining_seconds)
        else:
            eta = datetime.now() + timedelta(minutes=self.duration_minutes)

        return {
            "task_id": self.task_id,
            "status": "completed" if overall_progress >= 100 else "processing",
            "start_time": self.start_time.isoformat(),
            "progress_percent": round(overall_progress, 1),
            "items_processed": self.items_processed,
            "total_items": self.total_items,
            "current_stage": current_stage["name"],
            "stage_description": current_stage["description"],
            "stage_progress": round(stage_progress * 100, 1),
            "estimated_completion": eta.isoformat(),
            "elapsed_time_seconds": round(elapsed_seconds),
            "processing_type": self.processing_type,
            "dataset_size": self.dataset_size,
            "last_updated": datetime.now().isoformat(),
        }

    def process_batch(self):
        """Process a batch of items and update progress."""
        # Simulate variable processing speed (some batches take longer)
        base_delay = 1.0 / self.base_processing_speed * self.batch_size

        # Add some randomness to simulate real processing
        import random

        delay_multiplier = random.uniform(0.8, 1.2)
        actual_delay = base_delay * delay_multiplier

        time.sleep(min(actual_delay, 1.0))  # Cap at 1 second for responsiveness

        self.items_processed += self.batch_size

        # Check if we should move to next stage
        items_per_stage = self.total_items // len(self.PROCESSING_STAGES)
        expected_items_for_stage = (self.current_stage_index + 1) * items_per_stage

        if (
            self.items_processed >= expected_items_for_stage
            and self.current_stage_index < len(self.PROCESSING_STAGES) - 1
        ):
            self.current_stage_index += 1
            self.stage_start_time = time.time()
            logger.info("Processor %s: Moving to stage: %s", self.task_id, self.get_current_stage()["description"])

    def save_progress(self):
        """Save current progress to result file."""
        try:
            progress_data = self.calculate_progress()
            with open(self.result_file, "w") as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error("Processor %s: Error saving progress: %s", self.task_id, e)

    def cleanup(self):
        """Clean up result file after processing."""
        try:
            # Keep file for 5 minutes after completion for final reading
            time.sleep(300)
            if os.path.exists(self.result_file):
                os.remove(self.result_file)
                logger.info("Processor %s: Cleaned up result file", self.task_id)
        except Exception as e:
            logger.error("Processor %s: Error during cleanup: %s", self.task_id, e)


def run_data_processing(task_id: int, dataset_size: str, processing_type: str, duration_minutes: int, batch_size: int):
    """Main data processing function that runs in background thread."""
    processor = DataProcessor(task_id, dataset_size, processing_type, duration_minutes, batch_size)

    logger.info("Processor %s: Starting %s processing of %s dataset", task_id, processing_type, dataset_size)
    logger.info("Processor %s: Duration: %s minutes, Total items: %s", task_id, duration_minutes, processor.total_items)

    try:
        # Store processor reference
        active_tasks[task_id] = processor

        # Main processing loop
        while processor.items_processed < processor.total_items:
            processor.process_batch()
            processor.save_progress()

            # Break if we've exceeded our time limit (safety check)
            elapsed_minutes = (datetime.now() - processor.start_time).total_seconds() / 60
            if elapsed_minutes > duration_minutes * 1.2:  # 20% buffer
                logger.warning("Processor %s: Time limit exceeded, completing processing", task_id)
                break

        # Mark as completed
        processor.items_processed = processor.total_items
        processor.save_progress()

        logger.info("Processor %s: Processing completed successfully!", task_id)

    except Exception as e:
        logger.error("Processor %s: Error during processing: %s", task_id, e)
        # Save error state
        try:
            error_data = processor.calculate_progress()
            error_data["status"] = "failed"
            error_data["error"] = str(e)
            with open(processor.result_file, "w") as f:
                json.dump(error_data, f, indent=2)
        except Exception as e:
            pass

    finally:
        # Complete the async task
        success = app.complete_async_task(task_id)
        logger.info("Processor %s: Task completion: %s", task_id, "SUCCESS" if success else "FAILED")

        # Remove from active tasks
        active_tasks.pop(task_id, None)

        # Schedule cleanup
        cleanup_thread = threading.Thread(target=processor.cleanup, daemon=True)
        cleanup_thread.start()


@tool
def start_data_processing(
    dataset_size: str = "medium",
    processing_type: str = "data_analysis",
    duration_minutes: int = 30,
    batch_size: int = 100,
) -> str:
    """Start a long-running data processing task in the background.

    Args:
        dataset_size: Size of dataset to process ("small", "medium", "large", "huge")
        processing_type: Type of processing ("data_analysis", "ml_training", "data_cleaning", "feature_engineering")
        duration_minutes: How long the processing should take (default: 30 minutes)
        batch_size: Items to process per batch (affects update frequency)

    Returns:
        Status message with task details
    """

    # Validate inputs
    valid_sizes = ["small", "medium", "large", "huge"]
    valid_types = ["data_analysis", "ml_training", "data_cleaning", "feature_engineering"]

    if dataset_size.lower() not in valid_sizes:
        return f"❌ Invalid dataset_size. Choose from: {', '.join(valid_sizes)}"

    if processing_type.lower() not in valid_types:
        return f"❌ Invalid processing_type. Choose from: {', '.join(valid_types)}"

    if duration_minutes < 1 or duration_minutes > 180:
        return "❌ Duration must be between 1 and 180 minutes"

    # Start interactive task tracking
    task_metadata = {
        "dataset_size": dataset_size,
        "processing_type": processing_type,
        "duration_minutes": duration_minutes,
        "batch_size": batch_size,
    }

    task_id = app.add_async_task("data_processing", task_metadata)

    # Start background processing thread
    thread = threading.Thread(
        target=run_data_processing,
        args=(task_id, dataset_size, processing_type, duration_minutes, batch_size),
        daemon=True,
    )
    thread.start()

    return f"""🚀 **Data Processing Started!**

📊 **Task Details:**
  • Task ID: {task_id}
  • Dataset: {dataset_size.title()}
  • Type: {processing_type.replace("_", " ").title()}
  • Duration: {duration_minutes} minutes
  • Batch Size: {batch_size} items

📁 **Progress File:** `data_processing_results_{task_id}.json`

⏱️  **Status:** Processing will run for approximately {duration_minutes} minutes
📈 **Health:** Agent status now BUSY (check with get_health_status())

💡 **The agent remains fully interactive while processing!**
   Try asking: "What's the processing progress?" or any other questions.

🔍 **Monitor Progress:** Use get_processing_progress() or get_processing_progress({task_id})"""


@tool
def get_processing_progress(task_id: Optional[int] = None) -> str:
    """Get current progress of data processing task.

    Args:
        task_id: Specific task ID to check (optional - will find most recent if not provided)

    Returns:
        Detailed progress information
    """

    # Find result file
    result_file = None
    if task_id is not None:
        result_file = f"data_processing_results_{task_id}.json"
    else:
        # Find most recent result file
        result_files = [f for f in os.listdir(".") if f.startswith("data_processing_results_") and f.endswith(".json")]
        if result_files:
            # Sort by modification time, newest first
            result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            result_file = result_files[0]
            # Extract task_id from filename
            task_id = int(result_file.replace("data_processing_results_", "").replace(".json", ""))

    if not result_file or not os.path.exists(result_file):
        return """❌ **No Processing Task Found**

No active or recent data processing tasks detected.

💡 **Start a new task with:**
   `start_data_processing(dataset_size="medium", processing_type="data_analysis")`"""

    try:
        with open(result_file, "r") as f:
            progress = json.load(f)

        status = progress.get("status", "unknown")
        progress_percent = progress.get("progress_percent", 0)
        items_processed = progress.get("items_processed", 0)
        total_items = progress.get("total_items", 0)
        # current_stage value not used
        stage_description = progress.get("stage_description", "")
        stage_progress = progress.get("stage_progress", 0)
        elapsed_seconds = progress.get("elapsed_time_seconds", 0)

        # Format elapsed time
        elapsed_minutes = elapsed_seconds // 60
        elapsed_secs = elapsed_seconds % 60
        elapsed_str = f"{elapsed_minutes}m {elapsed_secs}s"

        # Calculate ETA
        eta_str = "Unknown"
        if "estimated_completion" in progress:
            try:
                eta = datetime.fromisoformat(progress["estimated_completion"])
                remaining = eta - datetime.now()
                if remaining.total_seconds() > 0:
                    remaining_minutes = remaining.total_seconds() // 60
                    eta_str = f"{int(remaining_minutes)} minutes"
                else:
                    eta_str = "Any moment now"
            except Exception:
                pass

        # Status-specific formatting
        if status == "completed":
            return f"""✅ **Processing Complete!**

📊 **Task #{task_id} Summary:**
  • Dataset: {progress.get("dataset_size", "unknown").title()}
  • Type: {progress.get("processing_type", "unknown").replace("_", " ").title()}
  • Items Processed: {items_processed:,} / {total_items:,}
  • Total Time: {elapsed_str}
  • Final Stage: {stage_description}

🎉 **Status:** Processing completed successfully!
📁 **Results:** Available in `{result_file}` (will be cleaned up in 5 minutes)"""

        elif status == "failed":
            error_msg = progress.get("error", "Unknown error")
            return f"""❌ **Processing Failed**

📊 **Task #{task_id} Status:**
  • Progress: {progress_percent}% complete
  • Items Processed: {items_processed:,} / {total_items:,}
  • Current Stage: {stage_description}
  • Error: {error_msg}
  • Elapsed Time: {elapsed_str}

🔧 **Try starting a new task with different parameters.**"""

        else:  # processing
            # Progress bar visualization
            bar_length = 20
            filled_length = int(bar_length * progress_percent / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            return f"""🔄 **Processing In Progress**

📊 **Task #{task_id} Status:**
  • Overall Progress: {progress_percent}% [{bar}]
  • Items: {items_processed:,} / {total_items:,} processed

🔧 **Current Stage:** {stage_description}
  • Stage Progress: {stage_progress}%

⏱️  **Timing:**
  • Elapsed: {elapsed_str}
  • ETA: ~{eta_str}

📈 **Details:**
  • Dataset: {progress.get("dataset_size", "unknown").title()}
  • Type: {progress.get("processing_type", "unknown").replace("_", " ").title()}

💡 **The agent remains fully responsive! Ask me anything else while we wait.**"""

    except Exception as e:
        return f"""❌ **Error Reading Progress**

Could not read progress file for task #{task_id}: {str(e)}

🔧 **Try:** Check if the task is still running or start a new task."""


@tool
def get_health_status() -> str:
    """Get current system health status and active task information."""
    status = app.get_current_ping_status()
    task_info = app.get_async_task_info()

    active_count = task_info.get("active_count", 0)
    running_jobs = task_info.get("running_jobs", [])

    if active_count == 0:
        return f"""🟢 **System Status: {status.value}**

✅ No background tasks running
💚 System ready for new data processing tasks

🚀 **Start a new task:**
   `start_data_processing(dataset_size="large", processing_type="ml_training")`"""
    else:
        jobs_text = ""
        for job in running_jobs:
            name = job.get("name", "unknown")
            duration = job.get("duration", 0)
            duration_str = f"{int(duration // 60)}m {int(duration % 60)}s" if duration > 60 else f"{int(duration)}s"
            jobs_text += f"\n  🔄 {name.replace('_', ' ').title()} (running {duration_str})"

        return f"""🟡 **System Status: {status.value}**

📊 **Active Tasks:** {active_count}{jobs_text}

💡 **Agent Interactivity:** Fully responsive despite background processing!
🔍 **Check Progress:** Use `get_processing_progress()` for detailed status"""


@tool
def list_available_options() -> str:
    """Show all available dataset sizes, processing types, and example configurations."""

    return """📋 **Available Processing Options**

**Dataset Sizes:**
  • `small` - ~500 items (faster for testing)
  • `medium` - ~2,000 items (balanced processing)
  • `large` - ~5,000 items (substantial workload)
  • `huge` - ~10,000 items (extensive processing)

**Processing Types:**
  • `data_analysis` - Statistical analysis and insights
  • `ml_training` - Machine learning model training
  • `data_cleaning` - Data validation and cleaning
  • `feature_engineering` - Feature extraction and transformation

⚙️ **Example Configurations:**

**Quick Test (2 minutes):**
```
start_data_processing(
    dataset_size="small",
    processing_type="data_analysis",
    duration_minutes=2
)
```

**Standard Analysis (15 minutes):**
```
start_data_processing(
    dataset_size="medium",
    processing_type="data_analysis",
    duration_minutes=15
)
```

**Heavy ML Training (60 minutes):**
```
start_data_processing(
    dataset_size="large",
    processing_type="ml_training",
    duration_minutes=60
)
```

💡 **Duration Range:** 1-180 minutes (default: 30 minutes)
⚡ **Batch Size:** 50-500 items per batch (default: 100)"""


# Create interactive agent
agent = Agent(tools=[start_data_processing, get_processing_progress, get_health_status, list_available_options])


@app.entrypoint
def agent_invocation(payload):
    """Main agent entrypoint."""
    user_message = payload.get(
        "prompt",
        "Hello! I can start long-running data processing tasks. Try: "
        "'Start processing a large dataset for ML training' or 'What are my options?'",
    )

    result = agent(user_message)

    return {"message": result.message, "demo": "Interactive Async Strands - Long-Running Data Processing"}


if __name__ == "__main__":
    app.logger.info("🤖 Interactive Async Strands Demo")
    app.logger.info("=" * 60)
    app.logger.info("🎯 Long-Running Data Processing with Real-Time Progress")
    app.logger.info("📊 Features: 30-min processing, file-based progress, agent interactivity")
    app.logger.info("🔄 Task Tracking: Proper async task lifecycle management")
    app.logger.info("")
    app.logger.info("🧪 Example Commands:")
    app.logger.info("")
    app.logger.info("1️⃣  **Start Processing:**")
    app.logger.info("curl -X POST http://localhost:8080/invocations \\")
    app.logger.info("  -H 'Content-Type: application/json' \\")
    app.logger.info('  -d \'{"prompt": "Start processing a medium dataset for data analysis"}\'')
    app.logger.info("")
    app.logger.info("2️⃣  **Check Progress (anytime during processing):**")
    app.logger.info("curl -X POST http://localhost:8080/invocations \\")
    app.logger.info("  -H 'Content-Type: application/json' \\")
    app.logger.info('  -d \'{"prompt": "What is the processing progress?"}\'')
    app.logger.info("")
    app.logger.info("3️⃣  **Test Interactivity (while processing):**")
    app.logger.info("curl -X POST http://localhost:8080/invocations \\")
    app.logger.info("  -H 'Content-Type: application/json' \\")
    app.logger.info('  -d \'{"prompt": "Tell me about the weather while we wait"}\'')
    app.logger.info("")
    app.logger.info("4️⃣  **Quick Test (2 minutes):**")
    app.logger.info("curl -X POST http://localhost:8080/invocations \\")
    app.logger.info("  -H 'Content-Type: application/json' \\")
    app.logger.info('  -d \'{"prompt": "Start a small dataset analysis for 2 minutes"}\'')
    app.logger.info("")
    app.logger.info("📊 **Expected Flow:**")
    app.logger.info("  • Health: HEALTHY → BUSY → HEALTHY")
    app.logger.info("  • Files: Progress saved every second to JSON")
    app.logger.info("  • Agent: Always responsive and interactive")
    app.logger.info("  • Processing: Realistic multi-stage simulation")
    app.logger.info("")
    app.logger.info("🚀 Starting server on http://localhost:8080")
    app.logger.info("=" * 60)

    app.run(port=8080)
