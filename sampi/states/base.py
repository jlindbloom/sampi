


from jlinops import isshape


class State:
    """Base class for representing unknowns in a statistical model.
    """

    def __init__(self, state_info):

    
        # Bind state_info
        self.state_info
        self.state_keys = self.state_info.keys()

        # Check that it looks right
        assert isinstance(state_info, dict), "state_info must be a dictionary!"
        for key in self.state_keys:
            assert isinstance(key, str), "keys of state_info must be strings!"
            assert isshape(self.state_info[key]) or (key == "scalar"), "values in state_info invalid"

        # Build dict for storing the unknowns
        self.states = {}
        for key in self.state_keys:
            self.states[key] = None


    def get_value(self, var_name):
        """Returns the value of a current unknown.
        """
        assert var_name in self.state_keys, "invalid variable name!"
        return self.states[var_name]







