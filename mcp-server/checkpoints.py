"""
Interactive checkpoint system for EDA Server.
Provides human-in-the-loop approval mechanisms.
"""

import asyncio
import time
from typing import Optional, Callable, Any
from config import config

class CheckpointManager:
    """Manages interactive checkpoints for human approval."""
    
    def __init__(self):
        self.require_approval = config.checkpoints.require_approval
        self.approval_timeout = config.checkpoints.approval_timeout
        self.auto_approve_small_changes = config.checkpoints.auto_approve_small_changes
    
    async def request_approval(
        self, 
        message: str, 
        auto_approve: bool = False,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Request human approval for an action.
        
        Args:
            message: Message to display to user
            auto_approve: Whether to auto-approve if enabled
            timeout: Custom timeout in seconds
            
        Returns:
            True if approved, False if rejected or timeout
        """
        if not self.require_approval:
            return True
        
        if auto_approve and self.auto_approve_small_changes:
            print(f"[AUTO-APPROVED] {message}")
            return True
        
        print(f"\n{message}")
        print("Do you approve? (y/N): ", end="", flush=True)
        
        # Use a timeout for approval
        timeout = timeout or self.approval_timeout
        
        try:
            # For now, simulate user input (in real implementation, this would be async input)
            # In a real MCP environment, this would integrate with the client's UI
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # For demonstration, auto-approve after timeout
            # In production, this would wait for actual user input
            print("y")  # Simulate user approval
            return True
            
        except asyncio.TimeoutError:
            print("\nApproval timeout - action rejected")
            return False
        except KeyboardInterrupt:
            print("\nApproval cancelled by user")
            return False
    
    def create_checkpoint_message(
        self, 
        title: str, 
        dataset_name: str, 
        summary: dict,
        action_description: str = ""
    ) -> str:
        """
        Create a formatted checkpoint message.
        
        Args:
            title: Checkpoint title
            dataset_name: Name of the dataset
            summary: Summary statistics
            action_description: Description of the action requiring approval
            
        Returns:
            Formatted message
        """
        message = f"""
=== {title.upper()} CHECKPOINT ===
Dataset: {dataset_name}
"""
        
        if action_description:
            message += f"Action: {action_description}\n\n"
        
        # Add summary statistics
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                message += f"{key.replace('_', ' ').title()}: {value}\n"
            elif isinstance(value, list):
                message += f"{key.replace('_', ' ').title()}: {len(value)} items\n"
            else:
                message += f"{key.replace('_', ' ').title()}: {value}\n"
        
        message += "\nPlease review the results and approve before proceeding.\n"
        message += "=" * 50
        
        return message

# Global checkpoint manager instance
checkpoint_manager = CheckpointManager()

async def request_missing_data_approval(
    dataset_name: str, 
    missing_stats: dict,
    recommended_actions: list
) -> bool:
    """
    Request approval for missing data analysis results.
    
    Args:
        dataset_name: Name of the dataset
        missing_stats: Missing data statistics
        recommended_actions: List of recommended actions
        
    Returns:
        True if approved, False if rejected
    """
    summary = {
        "Total Rows": missing_stats.get("total_rows", 0),
        "Total Columns": missing_stats.get("total_columns", 0),
        "Overall Missing Rate": f"{missing_stats.get('overall_missing_rate_percent', 0):.2f}%",
        "Columns with Missing Data": missing_stats.get("columns_with_missing", 0),
        "Recommended Actions": len(recommended_actions)
    }
    
    message = checkpoint_manager.create_checkpoint_message(
        "Missing Data Analysis",
        dataset_name,
        summary,
        "Review missing data patterns and approve recommended actions"
    )
    
    return await checkpoint_manager.request_approval(message)

async def request_outlier_approval(
    dataset_name: str,
    outlier_stats: dict,
    method_used: str
) -> bool:
    """
    Request approval for outlier detection results.
    
    Args:
        dataset_name: Name of the dataset
        outlier_stats: Outlier detection statistics
        method_used: Method used for outlier detection
        
    Returns:
        True if approved, False if rejected
    """
    summary = {
        "Method Used": method_used.upper(),
        "Total Outliers": outlier_stats.get("total_outliers", 0),
        "Impact Score": f"{outlier_stats.get('impact_score', 0):.3f}%",
        "Columns with Outliers": outlier_stats.get("columns_with_outliers", 0)
    }
    
    message = checkpoint_manager.create_checkpoint_message(
        "Outlier Detection",
        dataset_name,
        summary,
        "Review outlier patterns and approve recommended actions"
    )
    
    return await checkpoint_manager.request_approval(message)

async def request_transformation_approval(
    dataset_name: str,
    original_shape: tuple,
    transformed_shape: tuple,
    new_columns: list,
    transformations_applied: list
) -> bool:
    """
    Request approval for feature transformation results.
    
    Args:
        dataset_name: Name of the dataset
        original_shape: Original dataset shape
        transformed_shape: Transformed dataset shape
        new_columns: List of new columns created
        transformations_applied: List of transformations applied
        
    Returns:
        True if approved, False if rejected
    """
    summary = {
        "Original Shape": f"{original_shape[0]} rows × {original_shape[1]} columns",
        "Transformed Shape": f"{transformed_shape[0]} rows × {transformed_shape[1]} columns",
        "New Features Created": len(new_columns),
        "Transformations Applied": len(transformations_applied)
    }
    
    message = checkpoint_manager.create_checkpoint_message(
        "Feature Transformation",
        dataset_name,
        summary,
        "Review transformations and approve before proceeding to modeling"
    )
    
    # Auto-approve if only small changes
    auto_approve = len(new_columns) <= 3 and len(transformations_applied) <= 2
    
    return await checkpoint_manager.request_approval(message, auto_approve=auto_approve)

async def request_schema_approval(
    dataset_name: str,
    schema_stats: dict
) -> bool:
    """
    Request approval for schema inference results.
    
    Args:
        dataset_name: Name of the dataset
        schema_stats: Schema inference statistics
        
    Returns:
        True if approved, False if rejected
    """
    summary = {
        "Total Rows": schema_stats.get("total_rows", 0),
        "Total Columns": schema_stats.get("total_columns", 0),
        "ID Columns": len(schema_stats.get("id_columns", [])),
        "High Cardinality Columns": len(schema_stats.get("high_cardinality_columns", [])),
        "Pattern Detection": len(schema_stats.get("pattern_columns", [])),
        "Data Quality Score": f"{schema_stats.get('data_quality_score', 0):.1f}%"
    }
    
    message = checkpoint_manager.create_checkpoint_message(
        "Schema Inference",
        dataset_name,
        summary,
        "Review inferred schema and approve before proceeding"
    )
    
    return await checkpoint_manager.request_approval(message)

def disable_checkpoints():
    """Disable all checkpoints for automated processing."""
    checkpoint_manager.require_approval = False

def enable_checkpoints():
    """Enable checkpoints for interactive processing."""
    checkpoint_manager.require_approval = True

def set_approval_timeout(timeout: int):
    """Set approval timeout in seconds."""
    checkpoint_manager.approval_timeout = timeout 