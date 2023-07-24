import abc


class SubclassDispatcherMeta(abc.ABCMeta):
    """
    Metaclass whose primary object is to allow parent abstract class construct subclass object
    with a given string as identifier

    Example:
        # suppose Activation is ABC
        foo = Activation("relu")
        # foo is now ReLU object
    """

    _MISSING = object()

    @classmethod
    def __prepare__(metacls, name, bases):
        ns = super().__prepare__(name, bases)

        def __init_subclass__(clz=None):
            clz._registry[clz.__qualname__.lower()] = clz

        if ns.get("__init_subclass__", metacls._MISSING) is metacls._MISSING:
            ns["__init_subclass__"] = __init_subclass__

        return ns

    def __init__(cls, *args, **kwargs):
        cls._registry = {}
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        """
        When creating new instances, try to instantiate first.

        If failed due to abstract methods, find subclass with given name
        and instantiate.

        Raises error if both fails.
        """

        # 1. instantiation
        try:
            obj = super().__call__(*args, **kwargs)
            return obj
        except TypeError as e:
            # instantiation failed
            exc = e
            ...

        # 2. get name
        name = args[0] if args else SubclassDispatcherMeta._MISSING
        if name is SubclassDispatcherMeta._MISSING or not isinstance(name, str):
            # try to get from kwargs + get rid of the name argument
            name = kwargs.pop("name", SubclassDispatcherMeta._MISSING)
        else:
            # get rid of the name argument
            args = args[1:]

        if name is SubclassDispatcherMeta._MISSING:
            # did not provide name
            raise TypeError(f"Failed to instantiate {cls}: {exc}")

        constructor = cls._registry.get(name.lower())
        if constructor is None:
            # constructor unknown
            raise TypeError(f"Unknown constructor: {name.lower()}")

        # recursion, this constructor call will call this method again, and hopefully
        # successfully instantiate in the first part
        return constructor(*args, **kwargs)
