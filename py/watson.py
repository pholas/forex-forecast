import utils

log = utils.setup_logging(__name__)


class Watson:
    def __init__(self, wml_client, wos_client):
        self.wml_client = wml_client
        self.wos_client = wos_client

    def get_service_provider_by_name(self, service_provider_name):
        service_providers = self.wos_client.service_providers.list().result.service_providers
        log.debug("Service providers size: " + str(len(service_providers)))
        service_provider_id = None
        for service_provider in service_providers:
            if service_provider.entity.name == service_provider_name:
                service_provider_id = service_provider.metadata.id
                log.debug("Found the service_provider: {}".format(service_provider_id))
        return service_provider_id
