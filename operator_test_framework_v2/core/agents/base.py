"""
Base class for LLM-based autonomous agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum, auto


class AgentActionType(Enum):
    """Types of agent actions."""
    EXTRACT = auto()
    GENERATE = auto()
    VALIDATE = auto()
    REPAIR = auto()
    ANALYZE = auto()
    QUERY = auto()


@dataclass
class AgentAction:
    """An action taken by an agent.
    
    Attributes:
        action_type: Type of action
        description: Action description
        input_data: Input to the action
        output_data: Output from the action
        metadata: Additional metadata
    """
    action_type: AgentActionType
    description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Context for agent execution.
    
    Attributes:
        operator_spec: Operator specification
        history: Previous actions and results
        memory: Agent's working memory
        config: Agent configuration
    """
    operator_spec: Optional["OperatorSpec"] = None
    history: List[AgentAction] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def remember(self, key: str, value: Any) -> None:
        """Store something in memory."""
        ...
    
    def recall(self, key: str) -> Optional[Any]:
        """Retrieve from memory."""
        ...
    
    def add_action(self, action: AgentAction) -> None:
        """Add an action to history."""
        ...


class BaseAgent(ABC):
    """Abstract base class for LLM-based agents.
    
    Agents are autonomous components that use LLM reasoning to
    perform testing-related tasks. They maintain context and
    can take multiple actions to achieve their goals.
    
    Example:
        >>> class MyAgent(BaseAgent):
        ...     def execute(self, context):
        ...         # Agent logic using LLM
        ...         return result
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        name: str,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
    ):
        """Initialize the agent.
        
        Args:
            llm_client: LLM client for inference
            name: Agent name
            system_prompt: System prompt for LLM
            max_iterations: Maximum iterations for iterative tasks
        """
        ...
    
    @property
    def name(self) -> str:
        """Agent name."""
        ...
    
    @abstractmethod
    def execute(self, context: AgentContext) -> Any:
        """Execute the agent's task.
        
        Args:
            context: Agent context
            
        Returns:
            Task result
        """
        ...
    
    def llm_call(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        ...
    
    def structured_llm_call(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Call LLM with structured output.
        
        Args:
            prompt: Input prompt
            output_schema: Expected output schema
            temperature: Sampling temperature
            
        Returns:
            Structured output
        """
        ...
    
    def chain_of_thought(
        self,
        problem: str,
        reasoning_steps: int = 3,
    ) -> str:
        """Use chain-of-thought reasoning.
        
        Args:
            problem: Problem description
            reasoning_steps: Number of reasoning steps
            
        Returns:
            Reasoning and conclusion
        """
        ...
    
    def reflect(self, context: AgentContext) -> str:
        """Reflect on previous actions.
        
        Args:
            context: Agent context
            
        Returns:
            Reflection
        """
        ...


class ReActAgent(BaseAgent):
    """Agent using ReAct (Reasoning + Acting) pattern."""
    
    def __init__(
        self,
        llm_client: "LLMClient",
        name: str,
        tools: Dict[str, Callable],
    ):
        """Initialize ReAct agent.
        
        Args:
            llm_client: LLM client
            name: Agent name
            tools: Available tools/actions
        """
        ...
    
    @override
    def execute(self, context: AgentContext) -> Any:
        """Execute using ReAct loop."""
        ...
    
    def think(self, context: AgentContext, observation: str) -> str:
        """Reasoning step."""
        ...
    
    def act(self, context: AgentContext, thought: str) -> AgentAction:
        """Action step."""
        ...


class MultiAgentSystem:
    """System coordinating multiple agents."""
    
    def __init__(self):
        """Initialize multi-agent system."""
        self.agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent."""
        ...
    
    def orchestrate(
        self,
        task: str,
        context: AgentContext,
        agent_sequence: List[str],
    ) -> Any:
        """Orchestrate multiple agents.
        
        Args:
            task: Overall task description
            context: Shared context
            agent_sequence: Sequence of agent names to execute
            
        Returns:
            Final result
        """
        ...
