from typing import TypeVar, Generic, Callable, Optional
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class Option(Generic[T]):
    """Option monad implementation for handling None cases functionally."""
    value: Optional[T]
    
    @staticmethod
    def some(value: T) -> 'Option[T]':
        """Create an Option with a value."""
        return Option(value)
    
    @staticmethod
    def none() -> 'Option[T]':
        """Create an empty Option."""
        return Option(None)
    
    def map(self, f: Callable[[T], U]) -> 'Option[U]':
        """Apply a function to the contained value if it exists."""
        return Option(f(self.value)) if self.value is not None else Option(None)
    
    def flat_map(self, f: Callable[[T], 'Option[U]']) -> 'Option[U]':
        """Apply a function that returns an Option to the contained value if it exists."""
        return f(self.value) if self.value is not None else Option(None)
    
    def get_or_else(self, default: T) -> T:
        """Get the value or return a default if None."""
        return self.value if self.value is not None else default
    
    def is_some(self) -> bool:
        """Check if the Option contains a value."""
        return self.value is not None
    
    def is_none(self) -> bool:
        """Check if the Option is empty."""
        return self.value is None
    
    def or_else(self, other: 'Option[T]') -> 'Option[T]':
        """Return this Option if it has a value, otherwise return the other Option."""
        return self if self.is_some() else other