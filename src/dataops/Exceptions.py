class ExceptionWithIDMixin(object):
    def __init__(self, id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id


class ExceptionWithMsgMixin(object):
    def __init__(self, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message


class ModelInputsNotProvided(ExceptionWithIDMixin, ExceptionWithMsgMixin, Exception):
    def __init__(self, id, message):
        super(ExceptionWithIDMixin, self).__init__(id, message)


class WarningWithMsgMixin(object):
    def __init__(self, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message


class ModelInputsNotProvidedWarning(WarningWithMsgMixin, Warning):
    def __init__(self, message):
        super(WarningWithMsgMixin, self).__init__(message)
